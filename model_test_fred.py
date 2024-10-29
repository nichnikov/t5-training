import torch
from transformers import (GPT2Tokenizer,
                          T5ForConditionalGeneration)

from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import Trainer

device='cuda'
hf_model = "ai-forever/FRED-T5-large"
# hf_model = "ai-forever/FRED-T5-1.7B"

tokenizer = GPT2Tokenizer.from_pretrained("ai-forever/FRED-T5-1.7B", eos_token='</s>')
model = T5ForConditionalGeneration.from_pretrained('models_bss_fred_old')

texts = [
"Query: налог с дивидендов компании нерезиденту Document: С дивидендов и приравненных к ним доходов1 заплатите НДФЛ или налог на прибыль. Какую ставку применить, кто перечисляет налог и в какие сроки, смотрите в таблице. Relevant:",
"Query: налог с дивидендов компании нерезиденту Document: Лимит 5 млн руб. и прогрессивную шкалу ставок по пункту 1 статьи 224 НК применят к совокупности налоговых баз. Перечень таких налоговых баз резидентов – в таблице ниже. Relevant:	Ложь",
"Query: правительство повышает ставку ндфл до 18% Document: Лимит 5 млн руб. и прогрессивную шкалу ставок по пункту 1 статьи 224 НК применят к совокупности налоговых баз. Перечень таких налоговых баз резидентов – в таблице ниже. Relevant:	Правда",
"Query: правительство повышает ставку ндфл до 18% Document: Составляйте 6-НДФЛ нарастающим итогом: за I квартал, полугодие, девять месяцев и год. Для этого используйте сведения из регистров налогового учета по НДФЛ (п. 1.3, 1.4, 4.1 Порядка, утв. приказом ФНС от 15.10.2020 № ЕД-7-11/753@). Relevant:",
"Query: 6-ндфл сроки перечисления ндфл Document: Составляйте 6-НДФЛ нарастающим итогом: за I квартал, полугодие, девять месяцев и год. Для этого используйте сведения из регистров налогового учета по НДФЛ (п. 1.3, 1.4, 4.1 Порядка, утв. приказом ФНС от 15.10.2020 № ЕД-7-11/753@). Relevant:",
"Query: 6-ндфл сроки перечисления ндфл Document: Лимит 5 млн руб. и прогрессивную шкалу ставок по пункту 1 статьи 224 НК применят к совокупности налоговых баз. Перечень таких налоговых баз резидентов – в таблице ниже. Relevant:	Ложь",
"Query: налог на имущество для ип Document: На общей системе предприниматель платит налог на имущество по всем объектам, которые используются в бизнесе (ст. 400, 401 НК). Используемая в личных целях недвижимость облагается налогом по общим правилам (письмо Минфина от 25.12.2012 № 03-05-06-01/76). Relevant:",
"Query: налог на имущество для ип Document: Человек, который обязан сам рассчитать НДФЛ с каких-либо доходов и подать декларацию до 30 апреля следующего года, должен заплатить с них налог. Срок – не позднее 15 июля года, который следует за годом, когда получили эти доходы (п. 4 ст. 228 НК). Например, по доходам 2022 года НДФЛ нужно перечислить до 17 июля 2023 года включительно. Подробности – в рекомендации Срок сдачи декларации по НДФЛ. Relevant:",
"Query: срок уплаты ндфл по декларации 3-ндфл за 2022 Document: Человек, который обязан сам рассчитать НДФЛ с каких-либо доходов и подать декларацию до 30 апреля следующего года, должен заплатить с них налог. Срок – не позднее 15 июля года, который следует за годом, когда получили эти доходы (п. 4 ст. 228 НК). Например, по доходам 2022 года НДФЛ нужно перечислить до 17 июля 2023 года включительно. Подробности – в рекомендации Срок сдачи декларации по НДФЛ. Relevant:",
"Query: срок уплаты ндфл по декларации 3-ндфл за 2022 Document: На общей системе предприниматель платит налог на имущество по всем объектам, которые используются в бизнесе (ст. 400, 401 НК). Используемая в личных целях недвижимость облагается налогом по общим правилам (письмо Минфина от 25.12.2012 № 03-05-06-01/76). Relevant:",
"Query: срок уплаты ндфл по декларации 3-ндфл за 2022 ип на осно Document: Человек, который обязан сам рассчитать НДФЛ с каких-либо доходов и подать декларацию до 30 апреля следующего года, должен заплатить с них налог. Срок – не позднее 15 июля года, который следует за годом, когда получили эти доходы (п. 4 ст. 228 НК). Например, по доходам 2022 года НДФЛ нужно перечислить до 17 июля 2023 года включительно. Подробности – в рекомендации Срок сдачи декларации по НДФЛ. Relevant:",
"Query: срок уплаты ндфл по декларации 3-ндфл за 2022 ип на осно Document: В рамках камеральной проверки декларации НДС, в которой заявлена сумма к возмещению налога из бюджета, у организации могут затребовать документы, подтверждающие правомерность налоговых вычетов: счета-фактуры, первичные и иные документы (абз. 2 п. 8 ст. 88, ст. 165 и 172 НК). Четкого перечня таких документов в НК нет. Однако инспекторы при истребовании документов должны применять риск-ориентированный подход (письмо ФНС от 12.11.2020 № ЕА-4-15/18589). Relevant:",
"Query: какие документы может запросить налогов при возмещении ндс Document: В рамках камеральной проверки декларации НДС, в которой заявлена сумма к возмещению налога из бюджета, у организации могут затребовать документы, подтверждающие правомерность налоговых вычетов: счета-фактуры, первичные и иные документы (абз. 2 п. 8 ст. 88, ст. 165 и 172 НК). Четкого перечня таких документов в НК нет. Однако инспекторы при истребовании документов должны применять риск-ориентированный подход (письмо ФНС от 12.11.2020 № ЕА-4-15/18589). Relevant:",
"Query: какие документы может запросить налогов при возмещении ндс Document: Человек, который обязан сам рассчитать НДФЛ с каких-либо доходов и подать декларацию до 30 апреля следующего года, должен заплатить с них налог. Срок – не позднее 15 июля года, который следует за годом, когда получили эти доходы (п. 4 ст. 228 НК). Например, по доходам 2022 года НДФЛ нужно перечислить до 17 июля 2023 года включительно. Подробности – в рекомендации Срок сдачи декларации по НДФЛ. Relevant:",
"Query: возмещение коммунальных расходов арендатором Document: Порядок возмещения расходов на пособия зависит от того, когда они возникли. Для расходов до и после 2021 года действует свой порядок возмещения затрат. Образцы заявлений и порядок возмещения расходов смотрите ниже. Relevant:",
"Query: возмещение коммунальных расходов арендатором Document: В поле 110 «Сумма дохода, начисленная физическим лицам» отразите сумму начисленного дохода по всем физлицам нарастающим итогом с начала года. Relevant:"
]

for text in texts[:3]:
    input_ids = tokenizer.encode(text,  return_tensors="pt")
    outputs=model.generate(input_ids, eos_token_id=tokenizer.eos_token_id, max_length=64, early_stopping=True)
    print("outputs:", outputs)

    outputs_decode = tokenizer.decode(outputs[0][1:])
    print("outputs decode:", outputs_decode)

    outputs_logits=model.generate(input_ids, output_scores=True, return_dict_in_generate=True, 
                                  eos_token_id=tokenizer.eos_token_id, max_length=64, early_stopping=True)
    sigmoid_0 = torch.sigmoid(outputs_logits.scores[0][0])
    print("sigmoid:", sigmoid_0[2])