import mlflow
import torch
from cortex.plugins import ModelPlugin
from cortex._lib import exp
from torch.backends import cudnn
import torch.nn.functional as F
import numpy as np
from torchsummary import summary

from src.models.mix_match.utils import interleave
from src.models.utils import accuracy, MlflowLogger, WeightEMA, EarlyStopping, f1_score
from src.models.wideresnet import WideResNet_50_2


class MixMatchController(ModelPlugin):
    defaults = dict(
        data=dict(batch_size=dict(train=64, val=1, test=1), inputs=dict(inputs='images'), shuffle=True, skip_last_batch=True),
        train=dict(save_on_lowest=None, epochs=500, archive_every=100),
        optimizer=dict(optimizer='Adam', learning_rate=0.01, single_optimizer=True)
    )

    # TODO Not exactly the same batches, as in MixMatch
    def optimizer_step(self, retain_graph=False):
        super().optimizer_step(retain_graph)
        self.ema_optimizer.step()

    def routine(self, T: float = 0.75, alpha: float = 0.5, *args, **kwargs):
        """
        :param alpha: Parameter of beta distribution
        :param T: Sharpening temperature
        """
        if self.data.mode == 'test' or self.data.mode == 'val':
            targets_l = self.inputs('data.targets')
            inputs_l = self.inputs('data.images')

            self.losses.classifier = torch.tensor(0.0).to(exp.DEVICE)

            with torch.no_grad():
                outputs_l = self.nets.ema_classifier(inputs_l)

        else:
            targets_l = self.inputs('data_l.targets')
            inputs_l = self.inputs('data_l.images')
            inputs_u1, inputs_u2 = self.inputs('data_u.images')

            # Transform label to one-hot
            targets_l_oh = torch.zeros(self.data.batch_size['train'], self.get_dims('data.targets'))
            targets_l_oh[range(targets_l_oh.shape[0]), targets_l] = 1.0
            targets_l_oh = targets_l_oh.to(exp.DEVICE)

            with torch.no_grad():
                # compute guessed labels of unlabel samples
                outputs_u1 = self.nets.classifier(inputs_u1)
                outputs_u2 = self.nets.classifier(inputs_u2)
                p = (torch.softmax(outputs_u1, dim=1) + torch.softmax(outputs_u2, dim=1)) / 2
                pt = p ** (1 / T)
                targets_u = pt / pt.sum(dim=1, keepdim=True)
                targets_u = targets_u.detach()

            # mixup
            all_inputs = torch.cat([inputs_l, inputs_u1, inputs_u2], dim=0)
            all_targets = torch.cat([targets_l_oh, targets_u, targets_u], dim=0)

            l = np.random.beta(alpha, alpha)
            l = max(l, 1 - l)

            idx = torch.randperm(all_inputs.size(0))
            input_a, input_b = all_inputs, all_inputs[idx]
            target_a, target_b = all_targets, all_targets[idx]

            mixed_input = l * input_a + (1 - l) * input_b
            mixed_target = l * target_a + (1 - l) * target_b

            # interleave labeled and unlabed samples between batches to get correct batchnorm calculation
            mixed_input = list(torch.split(mixed_input, self.data.batch_size['train']))
            mixed_input = interleave(mixed_input, self.data.batch_size['train'])

            logits = [self.nets.classifier(mixed_input[0])]
            for input in mixed_input[1:]:
                logits.append(self.nets.classifier(input))

            # put interleaved samples back
            logits = interleave(logits, self.data.batch_size['train'])
            logits_l = logits[0]
            logits_u = torch.cat(logits[1:], dim=0)

            all_data_steps = len(self.data.loaders['data_l']['train'])
            Ll, Lu, w = self.loss(logits_l, mixed_target[:self.data.batch_size['train']],
                                  logits_u, mixed_target[self.data.batch_size['train']:],
                                  exp.INFO['data_steps'] / all_data_steps)

            # record loss
            self.losses.classifier = Ll + w * Lu

            # Write res
            self.add_results(losses_l=Ll.item())
            self.add_results(losses_u=Lu.item())
            self.add_results(w=w)

            with torch.no_grad():
                outputs_l = self.nets.classifier(inputs_l)

        # Cross-entropy
        with torch.no_grad():
            cross_entropy = self.criterion(outputs_l, targets_l)
            self.add_results(cross_entropy=cross_entropy)

            # Top-k accuracy
            labeled = 1 - targets_l.eq(-1).long()
            top1 = accuracy(outputs_l, targets_l, labeled, top=1)
            # top5 = accuracy(outputs_l, targets_l, labeled, top=5)
            self.add_results(acc_top1=top1)

            # F1-score
            f1 = f1_score(outputs_l, targets_l)
            self.add_results(f1_score=f1)

    def build(self, lambda_u: float = 12.5, ema_decay: float = 0.999, early_stopping: dict = None, pretrained=False,
              run_hash=None, log_to_mlflow=True, type_of_run=None, *args, **kwargs):
        """
        :param early_stopping: Parameters for early stopping
        :param type_of_run: Type of run to log to mlflow (as a tag): hyperparam_search, varying_number_of_labels, None
        :param run_hash: MD5 hash of hyperparameters string for effective hyperparameter search
        :param log_to_mlflow: Log run to mlflow
        :param ema_decay: Exponential moving average decay rate
        :param lambda_u: Unlabeled loss weight
        :param pretrained: Use pretrained on ImageNet encoder, freezing all the layers except last
        """
        cudnn.benchmark = True

        # Reset the data iterator and draw batch to perform shape inference.
        self.data.reset(mode='test', make_pbar=False)
        self.data.next()
        input_shape = self.get_dims('data.images')

        self.nets.classifier = WideResNet_50_2(num_classes=self.get_dims('data.targets'), pretrained=pretrained)
        self.nets.ema_classifier = WideResNet_50_2(num_classes=self.get_dims('data.targets'), pretrained=pretrained)
        print(summary(self.nets.classifier, input_shape))

        for param in self.nets.ema_classifier.parameters():
            param.detach_()

        self.ema_optimizer = WeightEMA(self.nets.classifier, self.nets.ema_classifier, alpha=exp.ARGS['model']['ema_decay'])
        if early_stopping is not None:
            self.early_stopping = EarlyStopping(**early_stopping)
        self.loss = SemiLoss(exp.ARGS['model']['lambda_u'], exp.ARGS['train']['epochs'])
        self.criterion = torch.nn.CrossEntropyLoss()

        if log_to_mlflow:
            MlflowLogger.start_run(exp.INFO['name'] + '_MixMatch')
            MlflowLogger.log_basic_run_params(input_shape)
            MlflowLogger.log_ssl_parameters()
            mlflow.set_tag('type_of_run', type_of_run)
            mlflow.log_param('ema_decay', exp.ARGS['model']['ema_decay'])
            mlflow.log_param('lambda_u', exp.ARGS['model']['lambda_u'])
            mlflow.log_param('T', exp.ARGS['model']['T'])
            mlflow.log_param('alpha', exp.ARGS['model']['alpha'])
            mlflow.log_param('run_hash', exp.ARGS['model']['run_hash'])
            mlflow.log_param('early_stopping', exp.ARGS['model']['early_stopping'])
            mlflow.log_param('pretrained', pretrained)

    def eval_loop(self):
        # Evaluation
        self.data.reset(mode='val')
        super().eval_loop()
        self.data.reset(mode='test')
        super().eval_loop()

        if not hasattr(self, 'early_stopping'):
            MlflowLogger.log_all_metrics('train')
            MlflowLogger.log_all_metrics('val')
            MlflowLogger.log_all_metrics('test')
        else:  # Early stopping
            self.early_stopping.on_epoch_end()

            # Mlflow Logging
            if exp.INFO['epoch'] >= exp.ARGS['train']['epochs']:
                epoch_to_log = self.early_stopping.best_epoch  # Last epoch - logging the best epoch (based on val)
            elif self.early_stopping.stopped_epoch is not None:
                epoch_to_log = self.early_stopping.stopped_epoch  # Stopped learning - logging the best epoch (based on val)
            else:
                epoch_to_log = exp.INFO['epoch']  # Logging current epoch

            MlflowLogger.log_all_metrics('train', epoch_to_log)
            MlflowLogger.log_all_metrics('val', epoch_to_log)
            MlflowLogger.log_all_metrics('test', epoch_to_log)

            if self.early_stopping.stopped_epoch is not None:
                mlflow.log_metric('stopped_epoch', self.early_stopping.stopped_epoch, step=exp.INFO['epoch'])
                exp.INFO['epoch'] = exp.ARGS['train']['epochs']

    def visualize(self):
        if exp.ARGS['data']['data']['split_labelled_and_unlabelled'] and self.data.mode == 'train':
            inputs_l = self.inputs('data_l.images')
            targets_l = self.inputs('data_l.targets')

            inputs_u = self.inputs('data_u.images')[0]
            targets_u = torch.full(targets_l.shape, -1, dtype=torch.long).to(targets_l.device)

            inputs = torch.cat([inputs_l, inputs_u])
            targets = torch.cat([targets_l, targets_u])
        else:
            inputs = self.inputs('data.images')
            targets = self.inputs('data.targets')

        inputs = inputs[:, :1, :, :]
        inputs *= 0.229
        inputs += 0.485

        self.add_image(F.adaptive_avg_pool2d(inputs, (64, 64)), name='Input', labels=targets)


class SemiLoss:
    def __init__(self, lambda_u, epochs):
        self.lambda_u = lambda_u
        self.epochs = epochs

    @staticmethod
    def linear_rampup(current, rampup_length):
        if rampup_length == 0:
            return 1.0
        else:
            current = np.clip(current / rampup_length, 0.0, 1.0)
            return float(current)

    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u) ** 2)

        return Lx, Lu, self.lambda_u * SemiLoss.linear_rampup(epoch, self.epochs)
