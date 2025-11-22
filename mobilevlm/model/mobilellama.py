from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaModel, LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from mobilevlm.model.mobilevlm import MobileVLMMetaModel, MobileVLMMetaForCausalLM
from transformers.utils import add_start_docstrings_to_model_forward

class MobileVLMConfig(LlamaConfig):
    model_type = "mobilevlm"


class MobileLlamaModel(MobileVLMMetaModel, LlamaModel):
    config_class = MobileVLMConfig

    def __init__(self, config: LlamaConfig):
        super(MobileLlamaModel, self).__init__(config)


class MobileLlamaForCausalLM(LlamaForCausalLM, MobileVLMMetaForCausalLM):
    config_class = MobileVLMConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = MobileLlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()  # Initialize weights and apply final processing

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        print(f"MobileLlamaForCausalLM forward called.")
        output_attentions = True
        # output_hidden_states = False
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict 
        print(f"Return dict: {return_dict}")
        print(f"Input IDs shape: {input_ids.shape if input_ids is not None else 'None'}")
        print(f"Images shape: {images.shape if images is not None else 'None'}")
        print(f"Use cache: {use_cache}")
        print(f"Output attentions: {output_attentions}")
        print(f"Output hidden states: {output_hidden_states}")
        if not use_cache:
            input_ids, attention_mask, past_key_values, inputs_embeds, labels, image_features, masks = \
                self.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels, images)
        
            # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
            dec_last_hidden_state, dec_hidden, dec_attn = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=False
            )
            
            text_mask, vision_mask = masks
            if text_mask is None:
                t_v_mask = None
            else:
                t_v_mask = vision_mask.unsqueeze(1).unsqueeze(-1).repeat(1,dec_attn[0].shape[1],1,vision_mask.shape[-1]) * text_mask.unsqueeze(1).unsqueeze(-2).repeat(1,dec_attn[0].shape[1],text_mask.shape[-1],1)
                del vision_mask
                del text_mask

            first_a = dec_attn[0]
            del dec_attn
            torch.cuda.empty_cache()

            hidden_states = dec_last_hidden_state
            logits = self.lm_head(hidden_states) # B,256,32000

            loss = None
            if labels is not None:
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = CrossEntropyLoss()
                shift_logits = shift_logits.view(-1, self.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model/pipeline parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels)

            # if not return_dict:
            #     output = (logits,) + outputs[1:]
            #     return (loss,) + output if loss is not None else output

            print(f"Loss: {loss}")
            print(f"Logits shape: {logits.shape}")
            print(f"Past key values length: {len(past_key_values) if past_key_values is not None else 'None'}")
            print(f"Hidden states shape: {hidden_states.shape}")
            print(f"Attentions shape: {first_a.shape if first_a is not None else 'None'}")
            print(f"t_v_mask shape: {t_v_mask.shape if t_v_mask is not None else 'None'}")
            
            return CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=past_key_values,
                hidden_states=None,
                attentions=None,
            ), image_features, first_a, t_v_mask
        
        else:
            temp_out = \
                self.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels, images)
            if len(temp_out)==5:
                input_ids, attention_mask, past_key_values, inputs_embeds, labels = temp_out
            else:
                input_ids, attention_mask, past_key_values, inputs_embeds, labels, _, _ = temp_out
            # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )
            print(f"Model outputs type: {type(outputs)}")
            print(f"Model outputs keys: {outputs.keys() if hasattr(outputs, 'keys') else 'N/A'}")
            print(f"Model outputs hidden states length: {len(outputs.hidden_states) if hasattr(outputs, 'hidden_states') else 'N/A'}")
            print(f"Last hidden state shape: {outputs.last_hidden_state.shape if hasattr(outputs, 'last_hidden_state') else 'N/A'}")
            print(f"Hidden states shape: {outputs.hidden_states[3].shape if hasattr(outputs, 'hidden_states') else 'N/A'}")
            print(f"Attentions shape: {outputs.attentions[0].shape if hasattr(outputs, 'attentions') else 'N/A'}")
            print(f"One of the past key values shape: {outputs.past_key_values[0][0].shape if hasattr(outputs, 'past_key_values') else 'N/A'}")
            hidden_states = outputs[0]
            logits = self.lm_head(hidden_states)

            loss = None
            if labels is not None:
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = CrossEntropyLoss()
                shift_logits = shift_logits.view(-1, self.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model/pipeline parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels)

            if not return_dict:
                output = (logits,) + outputs[1:]
                return (loss,) + output if loss is not None else output
            

            return CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
            }
        )
        return model_inputs
    
    

AutoConfig.register("mobilevlm", MobileVLMConfig)
AutoModelForCausalLM.register(MobileVLMConfig, MobileLlamaForCausalLM)