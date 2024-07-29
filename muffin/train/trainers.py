from torch import nn
from torch.utils.data.sampler import Sampler, RandomSampler, SequentialSampler
import os
import math
import torch
import wandb
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

from transformers import Trainer
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from torch import Tensor
from torch.nn import Module
from utils.utils import is_main_process

from muffin.eval.muffin_inference_logp import get_batch_logps, get_batch_logps_minicpm


class ChunckedRandomSampler(Sampler[int]):
    def __init__(self, data_source, chunk_size=5000) -> None:
        self.data_source = data_source
        self.chunk_size = chunk_size

    def __iter__(self):
        n = len(self.data_source)
        seed = int(torch.empty((), dtype=torch.int64).random_().item())
        print(f'Chuncked Random Sampler seed is {seed}')
        generator = torch.Generator()
        generator.manual_seed(seed)

        for st in torch.randperm(n // self.chunk_size, generator=generator).tolist():
            base = st * self.chunk_size
            for i in torch.randperm(self.chunk_size, generator=generator).tolist():
                yield base + i

        base = (n // self.chunk_size) * self.chunk_size
        for i in torch.randperm(n % self.chunk_size, generator=generator).tolist():
            yield base + i

    def __len__(self) -> int:
        return len(self.data_source)

class ZephyrTrainer(Trainer):
    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None:
            return None

        # Build the sampler.
        # return RandomSampler(self.train_dataset)
        return SequentialSampler(self.train_dataset)

        # if self.args.group_by_length:
        #     assert NotImplementedError
        # else:
        #     if len(self.train_dataset) >= 50_000_000:
        #         return ChunckedRandomSampler(self.train_dataset)
        #     else:
        #         # print(f'Data set size is :{len(self.train_dataset)}', flush=True)
        #         # return SequentialSampler(self.train_dataset)

        #         print(f'Shuffle Data set size is :{len(self.train_dataset)}', flush=True)
        #         return RandomSampler(self.train_dataset)

def forward_DPO(model, input_ids, labels, attention_mask, images, **kwargs):
    token_weighted = kwargs.pop('token_weighted', False)
    dpo_use_average = kwargs.pop('dpo_use_average', False)
    is_minicpm = kwargs.pop('is_minicpm', False)

    output = model(
        input_ids=input_ids,
        labels=labels,
        attention_mask=attention_mask,
        images=images,
        **kwargs
    )
    impl = get_batch_logps_minicpm if is_minicpm else get_batch_logps
    if token_weighted:
        token_log_prob = impl(
            output.logits, labels, return_per_token_logp=True)
        return token_log_prob
    else:
        log_prob, average_log_prob = impl(
            output.logits, labels, return_per_token_logp=False)
        if dpo_use_average:
            return average_log_prob
        return log_prob

def forward_KTO(model, input_ids, labels, attention_mask, images, **kwargs):
    token_weighted = kwargs.pop('token_weighted', False)
    kto_use_average = kwargs.pop('kto_use_average', False)
    is_minicpm = kwargs.pop('is_minicpm', False)

    output = model(
        input_ids=input_ids,
        labels=labels,
        attention_mask=attention_mask,
        images=images,
        **kwargs
    )
    impl = get_batch_logps_minicpm if is_minicpm else get_batch_logps
    if token_weighted:
        token_log_prob = impl(
            output.logits, labels, return_per_token_logp=True)
        return token_log_prob
    else:
        log_prob, average_log_prob = impl(
            output.logits, labels, return_per_token_logp=False)
        if kto_use_average:
            return average_log_prob
        return log_prob


def dpo_loss(policy_chosen_logps: torch.FloatTensor,
             policy_rejected_logps: torch.FloatTensor,
             reference_chosen_logps: torch.FloatTensor,
             reference_rejected_logps: torch.FloatTensor,
             beta: float,
             reference_free: bool = False) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """Compute the DPO loss for a batch of policy and reference model log probabilities.

    Args:
        policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
        policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
        reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
        reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
        beta: Temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5. We ignore the reference model as beta -> 0.
        reference_free: If True, we ignore the _provided_ reference model and implicitly use a reference model that assigns equal probability to all responses.

    Returns:
        A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
        The losses tensor contains the DPO loss for each example in the batch.
        The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps

    if reference_free:
        ref_logratios = 0

    logits = pi_logratios - ref_logratios

    losses = -F.logsigmoid(beta * logits)
    chosen_rewards = beta * (policy_chosen_logps -
                             reference_chosen_logps).detach()
    rejected_rewards = beta * \
        (policy_rejected_logps - reference_rejected_logps).detach()

    return losses, chosen_rewards, rejected_rewards

def entail_dpo_loss(policy_chosen_logps: torch.FloatTensor,
             policy_rejected_logps: torch.FloatTensor,
             reference_chosen_logps: torch.FloatTensor,
             reference_rejected_logps: torch.FloatTensor,
             beta: float,
             reference_free: bool = False,
             entail_score = torch.FloatTensor,) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """Compute the DPO loss for a batch of policy and reference model log probabilities.

    Args:
        policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
        policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
        reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
        reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
        beta: Temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5. We ignore the reference model as beta -> 0.
        reference_free: If True, we ignore the _provided_ reference model and implicitly use a reference model that assigns equal probability to all responses.
        data_dict: include entailment scores
    Returns:
        A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
        The losses tensor contains the DPO loss for each example in the batch.
        The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
    """
    # print(f"entail_score shape: {entail_score.shape}") # policy_chosen_logps shape: torch.Size([1])
    # print(f"policy_chosen_logps shape: {policy_chosen_logps.shape}") # entail_score shape: torch.Size([1, 3])
    entail_value = torch.tensor(entail_score[:, 1]).to(policy_chosen_logps.dtype)
    contra_value = torch.tensor(entail_score[:, 0]).to(policy_chosen_logps.dtype)
    
    # print(entail_value)
    
    entail_weight = torch.log(entail_value + 1)
    
    # weighted rejected answer
    policy_rejected_logps = entail_weight * policy_chosen_logps + (1 - entail_weight) * policy_rejected_logps
    reference_rejected_logps = entail_weight * reference_chosen_logps + (1 - entail_weight) * reference_rejected_logps
    
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps

    if reference_free:
        ref_logratios = 0

    logits = pi_logratios - ref_logratios

    losses = -F.logsigmoid(beta * logits)
    chosen_rewards = beta * (policy_chosen_logps -
                             reference_chosen_logps).detach()
    rejected_rewards = beta * \
        (policy_rejected_logps - reference_rejected_logps).detach()

    return losses, chosen_rewards, rejected_rewards, policy_rejected_logps, reference_rejected_logps


def kto_loss(policy_chosen_logps: torch.FloatTensor,
             policy_rejected_logps: torch.FloatTensor,
             reference_chosen_logps: torch.FloatTensor,
             reference_rejected_logps: torch.FloatTensor,
             beta: float,
             desirable_weight: float,
             undesirable_weight: float,
             reference_free: bool = False) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """Compute the DPO loss for a batch of policy and reference model log probabilities.

    Args:
        policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
        policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
        reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
        reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
        beta: Temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5. We ignore the reference model as beta -> 0.
        reference_free: If True, we ignore the _provided_ reference model and implicitly use a reference model that assigns equal probability to all responses.

    Returns:
        A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
        The losses tensor contains the DPO loss for each example in the batch.
        The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
    """
    # stop grad for kl
    # chosen or rejected batch has zero for mask logps
    chosen_kl = (policy_chosen_logps - reference_chosen_logps).mean().clamp(min=0).detach()
    rejected_kl = (policy_rejected_logps - reference_rejected_logps).mean().clamp(min=0).detach()
    kl = ((chosen_kl + rejected_kl) /2).clamp(min=0).detach()

    chosen_logratios = policy_chosen_logps - reference_chosen_logps
    rejected_logratios = policy_rejected_logps - reference_rejected_logps

    chosen_losses = 1 - F.sigmoid(beta * (chosen_logratios - kl))
    chosen_rewards = beta * chosen_logratios.detach()
    rejected_losses = 1 - F.sigmoid(beta * (kl - rejected_logratios))
    rejected_rewards = beta * rejected_logratios.detach()
    
    # TODO unfinished

    # pi_logratios = policy_chosen_logps - policy_rejected_logps
    # ref_logratios = reference_chosen_logps - reference_rejected_logps

    # if reference_free:
    #     ref_logratios = 0
    # import random
    # desirable_plug = 1.0
    # undesirable_plug = 1.0
    # prob = random.random()
    # if prob < 0.5:
    #     undesirable_plug = 0.
    # else:
    #     desirable_plug = 0.

    losses = desirable_weight * chosen_losses + undesirable_weight * rejected_losses

    return losses, chosen_rewards, rejected_rewards


def compute_weighted_logp(per_token_logp, labels, token_weight, use_average):
    loss_mask = (labels[:, 1:].clone() != -100)
    # print(f'compute wlogp {labels.shape} {loss_mask.shape}, {token_weight.shape}, {per_token_logp.shape}')
    weighted_mask = token_weight * loss_mask
    logp = (per_token_logp * weighted_mask).sum(-1)

    average_logp = logp / weighted_mask.sum(-1)
    if use_average:
        return average_logp
    return logp


def collect_preference_metrics(metrics, task,
                               chosen_rewards, rejected_rewards,
                               policy_rej_logp, policy_win_logp,
                               ref_rej_logp, ref_win_logp, reward_accuracies,
                               preprocess_func,
                               new_policy_rej_logp=None, new_ref_rej_logp=None
                               ):
    t = task
    metrics = {}
    metrics[f'rewards_{t}/chosen'] = preprocess_func(chosen_rewards)
    metrics[f'rewards_{t}/rejected'] = preprocess_func(rejected_rewards)
    metrics[f'logps_{t}/rejected'] = preprocess_func(policy_rej_logp)
    metrics[f'logps_{t}/chosen'] = preprocess_func(policy_win_logp)
    metrics[f'logps_{t}/ref_rejected'] = preprocess_func(ref_rej_logp)
    metrics[f'logps_{t}/ref_chosen'] = preprocess_func(ref_win_logp)
    metrics[f'rewards_{t}/accuracies'] = preprocess_func(
        reward_accuracies)
    metrics[f'rewards_{t}/margins'] = metrics[f'rewards_{t}/chosen'] - \
        metrics[f'rewards_{t}/rejected']
    if new_policy_rej_logp is not None:
        metrics[f'logps_{t}/new_rejected'] = preprocess_func(new_policy_rej_logp)
    if new_ref_rej_logp is not None:
        metrics[f'logps_{t}/ref_new_rejected'] = preprocess_func(new_ref_rej_logp)
    return metrics


def get_beta_and_logps(data_dict, model, args, is_minicpm=False, is_llava15=False):
    win_input_ids = data_dict.pop('win_input_ids')
    rej_input_ids = data_dict.pop('rej_input_ids')

    win_labels = data_dict.pop('win_labels')
    rej_labels = data_dict.pop('rej_labels')

    win_attention_mask = data_dict.pop('win_attention_mask')
    rej_attention_mask = data_dict.pop('rej_attention_mask')

    ref_win_avg_logp = data_dict.pop('ref_win_avg_logp')
    ref_rej_avg_logp = data_dict.pop('ref_rej_avg_logp')
    ref_win_logp = data_dict.pop('ref_win_logp')
    ref_rej_logp = data_dict.pop('ref_rej_logp')
    ref_win_per_token_logp = data_dict.pop('ref_win_per_token_logp')
    ref_rej_per_token_logp = data_dict.pop('ref_rej_per_token_logp')
    if args.dpo_use_average:
        ref_win_logp = ref_win_avg_logp
        ref_rej_logp = ref_rej_avg_logp

    beta = data_dict.pop('beta')
    if 'DPO' in args.task:
        images = data_dict.pop('images', None)
        if is_minicpm:
            # print(data_dict.keys())
            data_dict.pop('win_context_ids')
            data_dict.pop('rej_context_ids')
            concatenated_images = images
        else:
            if images is None or len(images) == 0:
                concatenated_images = None
            else:
                concatenated_images = torch.cat([images, images], dim=0)
                
    elif args.task == 'KTO':
        images = data_dict.pop('images', None)
        if is_minicpm:
            # print(data_dict.keys())
            data_dict.pop('win_context_ids')
            data_dict.pop('rej_context_ids')
            concatenated_images = images
        else:
            if images is None or len(images) == 0:
                concatenated_images = None
            else:
                concatenated_images = torch.cat([images, images], dim=0)

    concatenated_input_ids = data_dict.pop('concatenated_input_ids')
    concatenated_labels = data_dict.pop('concatenated_labels')
    concatenated_attention_mask = data_dict.pop('concatenated_attention_mask')
    concatenated_attention_mask = None

    win_token_weight = data_dict.pop('win_token_weight')
    rej_token_weight = data_dict.pop('rej_token_weight')
    concatenated_token_weight = data_dict.pop('concatenated_token_weight')
    
    # print(f'concatenated_input_ids: {concatenated_input_ids}')
    
    # print(f"concatenated_labels: {concatenated_labels}")

    if is_llava15:
        (
            _,
            _,
            _,
            _,
            concatenated_inputs_embeds,
            concatenated_labels
        ) = model.prepare_inputs_labels_for_multimodal(
            input_ids=concatenated_input_ids,
            position_ids=None,
            attention_mask=None,
            past_key_values=None,
            labels=concatenated_labels,
            images=concatenated_images,
        )
        if concatenated_inputs_embeds is not None:
            output = model.forward(
                inputs_embeds=concatenated_inputs_embeds,
                labels=None,
                **data_dict,
            )
        else:
            output = model.forward(
                input_ids=concatenated_input_ids,
                attention_mask=concatenated_attention_mask,
                labels=concatenated_labels,
                **data_dict,
            )
            
        log_prob, average_log_prob = get_batch_logps(
            output.logits, concatenated_labels, return_per_token_logp=False)
        if args.dpo_use_average:
            concatenated_logp = average_log_prob
        else:
            concatenated_logp = log_prob
    else:
        concatenated_logp = forward_DPO(model,
                                        concatenated_input_ids,
                                        concatenated_labels,
                                        concatenated_attention_mask,
                                        concatenated_images,
                                        token_weighted=args.dpo_token_weighted,
                                        dpo_use_average=args.dpo_use_average,
                                        is_minicpm=is_minicpm,
                                        **data_dict)
    win_size = win_input_ids.shape[0]
    rej_size = rej_input_ids.shape[0]
    assert win_size == rej_size

    if args.dpo_token_weighted:
        if is_llava15:
            raise NotImplementedError
        # print(f'compute_loss win {win_input_ids.shape} {win_labels.shape} {ref_win_per_token_logp.shape} {win_token_weight.shape}', flush=True)
        # print(f'compute_loss rej {rej_input_ids.shape} {rej_labels.shape} {ref_rej_per_token_logp.shape} {rej_token_weight.shape}', flush=True)
        # print(f'compute_loss cat {concatenated_input_ids.shape} {concatenated_labels.shape} {concatenated_logp.shape} {concatenated_token_weight.shape}', flush=True)

        # for i in range(len(ref_win_per_token_logp)):
        #     print(f'compuate loss {i} win_input_ids={win_input_ids[i]}\nwin_labels={win_labels[i]}\nwin_per_token_logp={ref_win_per_token_logp[i]}\nwin_token_weight={win_token_weight[i]}\n', flush=True)
        #     print(f'compuate loss {i} rej_input_ids={rej_input_ids[i]}\nrej_labels={rej_labels[i]}\nrej_per_token_logp={ref_rej_per_token_logp[i]}\nrej_token_weight={rej_token_weight[i]}\n', flush=True)
        ref_win_logp = compute_weighted_logp(
            ref_win_per_token_logp, win_labels, win_token_weight, args.dpo_use_average)
        ref_rej_logp = compute_weighted_logp(
            ref_rej_per_token_logp, rej_labels, rej_token_weight, args.dpo_use_average)
        concatenated_logp = compute_weighted_logp(
            concatenated_logp, concatenated_labels, concatenated_token_weight, args.dpo_use_average)

        if torch.any(torch.isnan(ref_win_logp)):
            print(f'ref_win_logp fail', flush=True)
            exit()
        if torch.any(torch.isnan(ref_rej_logp)):
            print(f'ref_rej_logp fail', flush=True)
            exit()
        if torch.any(torch.isnan(concatenated_logp)):
            print(f'concatenated_logp fail', flush=True)
            exit()

    policy_win_logp, policy_rej_logp = concatenated_logp.split(
        [win_size, rej_size])
    return policy_win_logp, policy_rej_logp, ref_win_logp, ref_rej_logp, beta

def get_beta_entail_and_logps(data_dict, model, args, is_minicpm=False, is_llava15=False):
    win_input_ids = data_dict.pop('win_input_ids')
    rej_input_ids = data_dict.pop('rej_input_ids')

    win_labels = data_dict.pop('win_labels')
    rej_labels = data_dict.pop('rej_labels')
    
    if 'entail_score' in data_dict:
        entail_score = data_dict.pop('entail_score')

    win_attention_mask = data_dict.pop('win_attention_mask')
    rej_attention_mask = data_dict.pop('rej_attention_mask')

    ref_win_avg_logp = data_dict.pop('ref_win_avg_logp')
    ref_rej_avg_logp = data_dict.pop('ref_rej_avg_logp')
    ref_win_logp = data_dict.pop('ref_win_logp')
    ref_rej_logp = data_dict.pop('ref_rej_logp')
    ref_win_per_token_logp = data_dict.pop('ref_win_per_token_logp')
    ref_rej_per_token_logp = data_dict.pop('ref_rej_per_token_logp')
    if args.dpo_use_average:
        ref_win_logp = ref_win_avg_logp
        ref_rej_logp = ref_rej_avg_logp

    beta = data_dict.pop('beta')
    if 'DPO' in args.task:
        images = data_dict.pop('images', None)
        if is_minicpm:
            # print(data_dict.keys())
            data_dict.pop('win_context_ids')
            data_dict.pop('rej_context_ids')
            concatenated_images = images
        else:
            if images is None or len(images) == 0:
                concatenated_images = None
            else:
                concatenated_images = torch.cat([images, images], dim=0)
                
    elif args.task == 'KTO':
        images = data_dict.pop('images', None)
        if is_minicpm:
            # print(data_dict.keys())
            data_dict.pop('win_context_ids')
            data_dict.pop('rej_context_ids')
            concatenated_images = images
        else:
            if images is None or len(images) == 0:
                concatenated_images = None
            else:
                concatenated_images = torch.cat([images, images], dim=0)

    concatenated_input_ids = data_dict.pop('concatenated_input_ids')
    concatenated_labels = data_dict.pop('concatenated_labels')
    concatenated_attention_mask = data_dict.pop('concatenated_attention_mask')
    concatenated_attention_mask = None

    win_token_weight = data_dict.pop('win_token_weight')
    rej_token_weight = data_dict.pop('rej_token_weight')
    concatenated_token_weight = data_dict.pop('concatenated_token_weight')
    
    # print(f'concatenated_input_ids: {concatenated_input_ids}')
    
    # print(f"concatenated_labels: {concatenated_labels}")

    if is_llava15:
        (
            _,
            _,
            _,
            _,
            concatenated_inputs_embeds,
            concatenated_labels
        ) = model.prepare_inputs_labels_for_multimodal(
            input_ids=concatenated_input_ids,
            position_ids=None,
            attention_mask=None,
            past_key_values=None,
            labels=concatenated_labels,
            images=concatenated_images,
        )
        if concatenated_inputs_embeds is not None:
            output = model.forward(
                inputs_embeds=concatenated_inputs_embeds,
                labels=None,
                **data_dict,
            )
        else:
            output = model.forward(
                input_ids=concatenated_input_ids,
                attention_mask=concatenated_attention_mask,
                labels=concatenated_labels,
                **data_dict,
            )
            
        log_prob, average_log_prob = get_batch_logps(
            output.logits, concatenated_labels, return_per_token_logp=False)
        if args.dpo_use_average:
            concatenated_logp = average_log_prob
        else:
            concatenated_logp = log_prob
    else:
        concatenated_logp = forward_DPO(model,
                                        concatenated_input_ids,
                                        concatenated_labels,
                                        concatenated_attention_mask,
                                        concatenated_images,
                                        token_weighted=args.dpo_token_weighted,
                                        dpo_use_average=args.dpo_use_average,
                                        is_minicpm=is_minicpm,
                                        **data_dict)
    win_size = win_input_ids.shape[0]
    rej_size = rej_input_ids.shape[0]
    assert win_size == rej_size

    if args.dpo_token_weighted:
        if is_llava15:
            raise NotImplementedError
        # print(f'compute_loss win {win_input_ids.shape} {win_labels.shape} {ref_win_per_token_logp.shape} {win_token_weight.shape}', flush=True)
        # print(f'compute_loss rej {rej_input_ids.shape} {rej_labels.shape} {ref_rej_per_token_logp.shape} {rej_token_weight.shape}', flush=True)
        # print(f'compute_loss cat {concatenated_input_ids.shape} {concatenated_labels.shape} {concatenated_logp.shape} {concatenated_token_weight.shape}', flush=True)

        # for i in range(len(ref_win_per_token_logp)):
        #     print(f'compuate loss {i} win_input_ids={win_input_ids[i]}\nwin_labels={win_labels[i]}\nwin_per_token_logp={ref_win_per_token_logp[i]}\nwin_token_weight={win_token_weight[i]}\n', flush=True)
        #     print(f'compuate loss {i} rej_input_ids={rej_input_ids[i]}\nrej_labels={rej_labels[i]}\nrej_per_token_logp={ref_rej_per_token_logp[i]}\nrej_token_weight={rej_token_weight[i]}\n', flush=True)
        ref_win_logp = compute_weighted_logp(
            ref_win_per_token_logp, win_labels, win_token_weight, args.dpo_use_average)
        ref_rej_logp = compute_weighted_logp(
            ref_rej_per_token_logp, rej_labels, rej_token_weight, args.dpo_use_average)
        concatenated_logp = compute_weighted_logp(
            concatenated_logp, concatenated_labels, concatenated_token_weight, args.dpo_use_average)

        if torch.any(torch.isnan(ref_win_logp)):
            print(f'ref_win_logp fail', flush=True)
            exit()
        if torch.any(torch.isnan(ref_rej_logp)):
            print(f'ref_rej_logp fail', flush=True)
            exit()
        if torch.any(torch.isnan(concatenated_logp)):
            print(f'concatenated_logp fail', flush=True)
            exit()

    policy_win_logp, policy_rej_logp = concatenated_logp.split(
        [win_size, rej_size])
    return policy_win_logp, policy_rej_logp, ref_win_logp, ref_rej_logp, beta, entail_score


class LLaVA15DPOTrainer(ZephyrTrainer):

    def compute_loss(self, model: Module, inputs: dict, return_outputs=False):
        if self.args.past_index >= 0:
            raise NotImplementedError

        def gather_and_do_mean(x):
            return self._nested_gather(x.mean()).mean().item()

        data_dict = inputs
        policy_win_logp, policy_rej_logp, ref_win_logp, ref_rej_logp, beta = get_beta_and_logps(
            data_dict, model, self.args, is_llava15=True)

        losses, chosen_rewards, rejected_rewards = dpo_loss(policy_win_logp,
                                                            policy_rej_logp,
                                                            ref_win_logp,
                                                            ref_rej_logp,
                                                            beta=beta)
        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        SFT_weight = float(os.environ.get('SFT_weight', 0.0))
        DPO_weight = float(os.environ.get('DPO_weight', 1.0))
        loss = DPO_weight * losses.mean() - SFT_weight * policy_win_logp.mean()

        t = 'train' if model.training else 'test'
        metrics = {}
        metrics = collect_preference_metrics(metrics, t, chosen_rewards, rejected_rewards,
                                             policy_rej_logp, policy_win_logp,
                                             ref_rej_logp, ref_win_logp, reward_accuracies,
                                             gather_and_do_mean)
        self.log(metrics)

        return loss

class LLaVA15EntailDPOTrainer(ZephyrTrainer):

    def compute_loss(self, model: Module, inputs: dict, return_outputs=False):
        if self.args.past_index >= 0:
            raise NotImplementedError

        def gather_and_do_mean(x):
            return self._nested_gather(x.mean()).mean().item()

        data_dict = inputs
        policy_win_logp, policy_rej_logp, ref_win_logp, ref_rej_logp, beta, entail_score = get_beta_entail_and_logps(
            data_dict, model, self.args, is_llava15=True)

        losses, chosen_rewards, rejected_rewards, new_policy_rej_logp, new_ref_rej_logp  = entail_dpo_loss(policy_win_logp,
                                                            policy_rej_logp,
                                                            ref_win_logp,
                                                            ref_rej_logp,
                                                            beta=beta,
                                                            entail_score=entail_score)
        
        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        SFT_weight = float(os.environ.get('SFT_weight', 0.0))
        DPO_weight = float(os.environ.get('DPO_weight', 1.0))
        loss = DPO_weight * losses.mean() - SFT_weight * policy_win_logp.mean()

        t = 'train' if model.training else 'test'
        metrics = {}
        metrics = collect_preference_metrics(metrics, t, chosen_rewards, rejected_rewards,
                                             policy_rej_logp, policy_win_logp,
                                             ref_rej_logp, ref_win_logp, reward_accuracies,
                                             gather_and_do_mean, new_policy_rej_logp, new_ref_rej_logp)
        self.log(metrics)

        return loss


class LLaVA15KTOTrainer(ZephyrTrainer):

    def compute_loss(self, model: Module, inputs: dict, return_outputs=False):
        if self.args.past_index >= 0:
            raise NotImplementedError

        def gather_and_do_mean(x):
            return self._nested_gather(x.mean()).mean().item()

        data_dict = inputs
        policy_win_logp, policy_rej_logp, ref_win_logp, ref_rej_logp, beta = get_beta_and_logps(
            data_dict, model, self.args, is_llava15=True)

        losses, chosen_rewards, rejected_rewards = kto_loss(policy_win_logp,
                                                            policy_rej_logp,
                                                            ref_win_logp,
                                                            ref_rej_logp,
                                                            beta=beta,
                                                            desirable_weight=1.0,
                                                            undesirable_weight=1.0)
        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        SFT_weight = float(os.environ.get('SFT_weight', 0.0))
        DPO_weight = float(os.environ.get('DPO_weight', 1.0))
        loss = DPO_weight * losses.mean() - SFT_weight * policy_win_logp.mean()

        t = 'train' if model.training else 'test'
        metrics = {}
        metrics = collect_preference_metrics(metrics, t, chosen_rewards, rejected_rewards,
                                             policy_rej_logp, policy_win_logp,
                                             ref_rej_logp, ref_win_logp, reward_accuracies,
                                             gather_and_do_mean)
        self.log(metrics)

        return loss

