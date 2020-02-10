#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import torch.nn as nn

from transformers import (BertConfig,
                          BertForSequenceClassification, BertTokenizer,
                          RobertaConfig,
                          RobertaForSequenceClassification,
                          RobertaTokenizer,
                          XLMConfig, XLMForSequenceClassification,
                          XLMTokenizer, XLNetConfig,
                          XLNetForSequenceClassification,
                          XLNetTokenizer,
                          DistilBertConfig,
                          DistilBertForSequenceClassification,
                          DistilBertTokenizer)
from transformers import AdamW, WarmupLinearSchedule
from sklearn.metrics import classification_report, f1_score
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ExponentialLR
# from apex import amp
from tqdm import tqdm


MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer)
}

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, XLNetConfig, XLMConfig, 
                                                                                RobertaConfig, DistilBertConfig)), ())


class Model:
    """Enhanced Sequential Inference Model (ESIM) for natural language inference.
    """
    def __init__(self, args):
        """Model initialization.
        """
        self.args = args
        self.logger = args.logger

        self._build_model()
        self.model.to(args.device)

        self.optimizer = self._get_optimizer(self._group_parameters(self.model))
        self.scheduler = self._get_scheduler(self.optimizer)

        # Amp: Automatic Mixed Precision
        if self.args.fp16:
            self.model, self.optimizer = amp.initialize(self.model,
                                                        self.optimizer,
                                                        opt_level=args.fp16_opt_level)
            self.logger.info("- Automatic Mixed Precision (AMP) is used.")
        else:
            self.logger.info("- NO Automatic Mixed Precision (AMP) :/")

        # multi-gpu training (should be after apex fp16 initialization)
        if args.n_gpu > 1:
            self.logger.info("- Let's use {} GPUs !".format(torch.cuda.device_count()))
            self.model = nn.DataParallel(self.model)
        else:
            self.logger.info("- Train the model on single GPU :/")

        # tensorboard
        if args.write_summary:
            self.logger.info("- Let's use tensorboard on local rank {} device :)".format(args.local_rank))
            self.writer = SummaryWriter(self.args.summary_path)

    def _build_model(self):
        """Build model.
        """
        model_type = self.args.model_type.lower()
        self.config_class, self.model_class, self.tokenizer_class = MODEL_CLASSES[model_type]
        if self.args.use_pretrained:
            self.load_weights(self.args.checkpoint)
        else:
            self._load_from_library(self.args)

    def _load_from_library(self, args):
        """Initialize ESIM model paramerters.
        """
        self.logger.info("- Downloading model...")
        config = self.config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                                   num_labels=args.num_labels,
                                                   finetuning_task=args.task_name)
        self.tokenizer = self.tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name
                                                              else args.model_name_or_path,
                                                              do_lower_case=args.do_lower_case)
        self.model = self.model_class.from_pretrained(args.model_name_or_path,
                                                      from_tf=bool('.ckpt' in args.model_name_or_path),
                                                      config=config)

    def _group_parameters(self, model):
        """Specify which parameters do weight decay and which not.
        """
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params':
                [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': self.args.weight_decay},
            {'params':
                [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0}
            ]
        return optimizer_grouped_parameters

    def _get_optimizer(self, optimizer_grouped_parameters):
        """Get optimizer for model training.
        """
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.args.learning_rate,
                          eps=self.args.adam_epsilon)
        return optimizer

    def _get_scheduler(self, optimizer):
        """Get scheduler for adjusting learning rate.
        """
        if self.args.scheduler == 'warmup':
            scheduler = WarmupLinearSchedule(optimizer,
                                             warmup_steps=self.args.warmup_steps,
                                             t_total=self.args.num_epochs)
        elif self.args.scheduler == 'exponential':
            scheduler = ExponentialLR(optimizer, 0.95)
        return scheduler

    def load_weights(self, checkpoint):
        """Load pre-trained model weights.
        """
        self.logger.info("- Load pre-trained model from: {}".format(checkpoint))
        self.model = self.model_class.from_pretrained(checkpoint)
        self.tokenizer = self.tokenizer_class.from_pretrained(checkpoint,
                                                              do_lower_case=self.args.do_lower_case)
        self.model.to(self.args.device)
        return self.model, self.tokenizer

    def load_transformer(self, weights):
        """Load pre-trained model weights.
        """
        # Take care of distributed/parallel training
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        model_type = self.args.model_type.lower()
        if model_type == 'bert':
            model_to_save.bert.load_state_dict(weights.state_dict())
        elif model_type == 'xlnet':
            model_to_save.transformer.load_state_dict(weights.state_dict())
        elif model_type == 'distilbert':
            model_to_save.distilbert.load_state_dict(weights.state_dict())
        else:
            raise Exception("Unknow model type!")

    def save_model(self, output_path):
        """Save model's weights.
        """
        # Take care of distributed/parallel training
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_save.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        torch.save(self.args, os.path.join(output_path, 'training_args.bin'))
        self.logger.info("- model, tokenzier and args is saved at: {}".format(output_path))

    def loss_batch(self, inputs, optimizer=None, step=None):
        """Calculate loss on a single batch of data.
        """
        if optimizer:
            assert step is not None
        outputs = self.model(**inputs)
        loss, logits = outputs[0], outputs[1]

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        if optimizer is not None:
            if self.args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if (step + 1) % self.args.gradient_accumulation_steps == 0:
                if self.args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer),
                                                   self.args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                   self.args.max_grad_norm)
                optimizer.step()  # update model parameters
                optimizer.zero_grad()  # clean all gradients

        return loss.item(), logits.detach()

    def train_epoch(self, train_dataloader, optimizer, epoch):
        """Train the model for one single epoch.
        """
        self.model.train()  # set the model to training mode
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")

        train_loss = 0.0
        for i, batch in enumerate(epoch_iterator):
            batch = tuple(t.to(self.args.device) for t in batch)
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'labels':         batch[3]}

            # XLM, DistilBERT and RoBERTa don't use segment_ids
            if self.args.model_type != 'distilbert':
                inputs['token_type_ids'] = batch[2] if self.args.model_type in ['bert', 'xlnet'] else None

            batch_loss, _ = self.loss_batch(inputs,
                                            optimizer=optimizer,
                                            step=i)
            train_loss += batch_loss

            if self.writer:
                self.writer.add_scalar('batch_loss', batch_loss, epoch*len(train_dataloader) + i + 1)

        # compute the average loss (batch loss)
        epoch_loss = train_loss / len(train_dataloader)

        # update scheduler
        self.scheduler.step()

        return epoch_loss

    def evaluate(self, eval_dataloader, print_report=False):
        """Evaluate the model.
        """
        self.model.eval()  # set the model to evaluation mode
        with torch.no_grad():
            pred_class, label_class = [], []
            eval_loss, eval_corrects = 0.0, 0.0
            for _, batch in enumerate(tqdm(eval_dataloader, desc="Iteration")):
                batch = tuple(t.to(self.args.device) for t in batch)
                inputs = {'input_ids':      batch[0],
                          'attention_mask': batch[1],
                          'labels':         batch[3]}
                if self.args.model_type != 'distilbert':
                    inputs['token_type_ids'] = batch[2] if self.args.model_type in ['bert', 'xlnet'] else None

                batch_loss, outputs = self.loss_batch(inputs, optimizer=None)
                _, preds = torch.max(outputs, 1)  # preds: [batch_size]

                # save predictions
                pred_class += preds.tolist()
                label_class += batch[3].tolist()

                # update loss & accuracy
                eval_loss += batch_loss
                eval_corrects += torch.sum(preds == (batch[3])).double()

            avg_loss = eval_loss / len(eval_dataloader)
            avg_acc = eval_corrects / len(eval_dataloader.dataset)

        macro_f1 = f1_score(label_class, pred_class, average='macro')

        if print_report:
            self.logger.info('\n')
            # target_names = ['False', 'Partly Ture', 'True']
            self.logger.info(classification_report(label_class, pred_class))
        return avg_loss, avg_acc, macro_f1

    def fit(self, train_dataloader, eval_dataloader):
        """Model training and evaluation.
        """
        best_f1 = 0.
        num_epochs = self.args.num_epochs

        for epoch in range(num_epochs):
            self.logger.info('Epoch {}/{}'.format(epoch + 1, num_epochs))

            # training
            train_loss = self.train_epoch(train_dataloader, self.optimizer, epoch)
            self.logger.info("Traing Loss: {}".format(train_loss))

            # evaluation, only on the master node
            eval_loss, eval_acc, macro_f1 = self.evaluate(eval_dataloader, True)
            self.logger.info("Evaluation:")
            self.logger.info("- loss: {}".format(eval_loss))
            self.logger.info("- acc: {}".format(eval_acc))
            self.logger.info("- macro F1: {}".format(macro_f1))

            # monitor loss and accuracy
            if self.writer:
                self.writer.add_scalar('epoch_loss', train_loss, epoch)
                self.writer.add_scalar('eval_loss', eval_loss, epoch)
                self.writer.add_scalar('eval_acc', eval_acc, epoch)
                self.writer.add_scalar('lr', self.scheduler.get_lr()[0], epoch)

            # save the model
            if macro_f1 >= best_f1:
                best_f1 = macro_f1
                self.logger.info("New best score!")
                self.save_model(self.args.model_dir)

    def test(self, test_dataloader):
        """Test the model on unlabeled dataset.
        """
        self.model.eval()  # set the model to evaluation mode
        pred_class = []
        for _, batch in enumerate(tqdm(test_dataloader, desc="Iteration")):
            batch = tuple(t.to(self.args.device) for t in batch)
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'labels':         None}
            if self.args.model_type != 'distilbert':
                inputs['token_type_ids'] = batch[2] if self.args.model_type in ['bert', 'xlnet'] else None
            outputs = self.model(**inputs)
            _, preds = torch.max(outputs[0].detach(), 1)  # preds: [batch_size]
            pred_class += preds.tolist()
        return pred_class
