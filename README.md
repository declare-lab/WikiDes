# WikiDes: A Wikipedia-Based Dataset for Generating Short Descriptions from Paragraphs

We present **Wikides** (Wikipedia descriptions), a dataset for generating descriptions of Wikidata from Wikipedia paragraphs. This is a problem of both extreme summarization and indicative summarization. However, the generated descriptions from two-phase summarization makes the problem turn more to informative summarization. In addition, the dataset can use for other problems of NLP:
* Title generation
* Text classification based on instances (topics)
* Extract instances (topics) from text

# Dataset 

The data contain over 80k samples in the file **collected_data.json**. Please get the file here: **https://github.com/declare-lab/WikiDes/blob/main/dataset/collected_data.zip**.  

## Dataset Description

The collected dataset (e.g. **collected_data.json**) contains these fields:
* **wikidata_id**:  the identifier of a Wikidata item, https://www.wikidata.org/wiki/Wikidata:Identifiers.
* **label**: the label of a Wikidata item or the Wikipedia article title, https://www.wikidata.org/wiki/Help:Label.
* **description**: the description of a Wikidata item or the **gold description**, https://www.wikidata.org/wiki/Help:Description.
* **instances**:  a list of instances (P31) of a Wikidata item, https://www.wikidata.org/wiki/Help:Basic_membership_properties#instance_of_(P31). They can be considered as the baseline descriptions. In the experiment, we take the first element of this list as the baseline description.
* **subclasses**: a list of subclasses (P279) of a Wikidata. Property P279 is used to state that all the instances of one class are instances of another, https://www.wikidata.org/wiki/Help:Basic_membership_properties. *We do not use this list in the experiment but it can combine with instances or replace the role of instances as the baseline descriptions*.
* **aliases**: They are alternative names for the label of a Wikidata item, https://www.wikidata.org/wiki/Help:Aliases.
* **first_paragraph**: the first paragraph of a Wikipedia article that is linked to a Wikidata item.
* **first_sentence**: the first sentence of the the first paragraph. We use package NLTK punkt, https://www.nltk.org/_modules/nltk/tokenize/punkt.html for the extraction.

### A sample from the collected dataset

```
{
    "wikidata_id": "Q65293712", 
    "label": "Lepisma saccharina", 
    "description": "small, wingless insect in the order Thysanura", 
    "instances": [
        [
            "Q16521", 
            "taxon"
        ]
    ], 
    "subclasses": [
        [
            "Q219174", 
            "pest"
        ]
    ], 
    "aliases": [
        "Lepisma saccharina", 
        "fishmoth", 
        "Silverfish"
    ], 
    "first_paragraph": "The silverfish (Lepisma saccharinum) is a species of small, primitive, wingless insect in the order Zygentoma (formerly Thysanura). Its common name derives from the insect's silvery light grey colour, combined with the fish-like appearance of its movements. The scientific name (L. saccharinum) indicates that the silverfish's diet consists of carbohydrates such as sugar or starches. While the common name silverfish is used throughout the global literature to refer to various species of Zygentoma, the Entomological Society of America restricts use of the term solely for Lepisma saccharinum.", 
    "first_sentence": "The silverfish (Lepisma saccharinum) is a species of small, primitive, wingless insect in the order Zygentoma (formerly Thysanura)."
}
```

# Training Process to Generate Descriptions

There are 2 phases of training, **description generation and candidate ranking**. 

## Phrase 1. Description generation 
We consider Wikidata instances (https://www.wikidata.org/wiki/Help:Basic_membership_properties#instance_of_(P31)) as topics of samples. The data distribution is training set ~ 80%, validation set ~ 10%, and test set ~ 10%. We use first 256 tokens in Wikipedia first paragraphs as the documents in the training.
* **topic-exclusive split (diff)**: The data is split in different topics. The training set will have different topics from validation and test sets. The distribution of training, validation, and test sets is 65,772/7,820/7,827. Please get the file here: https://github.com/declare-lab/WikiDes/tree/main/dataset/phase1/diff.
* **topic-independent split (random)**: The data is split in random topics. All sets will have random topics. The distribution of training, validation, and test sets is 68,296/8,540/8,542.  Note that we did not filter empty Wikidata instances in this split.  Please get the file here: https://github.com/declare-lab/WikiDes/tree/main/dataset/phase1/random.

### A detail sample of Phase 1

```
{"wikidata_id": "Q55135146", "label": "Xyleborus intrusus", "source": "Xyleborus intrusus is a species of typical bark beetle in the family Curculionidae. It is found in North America.", "target": "species of insect", "baseline_candidates": ["taxon"]}
```

* **wikidata_id**: the identifier of a Wikidata item, https://www.wikidata.org/wiki/Wikidata:Identifiers.
* **label**: the label of a Wikidata item or the Wikipedia article title, https://www.wikidata.org/wiki/Help:Label.
* **source**: the first paragraph of a Wikipedia article.
* **target**: the description of a Wikidata item or the **gold description**, https://www.wikidata.org/wiki/Help:Description.
* **baseline_candidates**: a list of instances (P31) of a Wikidata item, https://www.wikidata.org/wiki/Help:Basic_membership_properties#instance_of_(P31). They are considered as the baseline descriptions.


### Some samples of Phase 1:
```
{"wikidata_id": "Q55135146", "label": "Xyleborus intrusus", "source": "Xyleborus intrusus is a species of typical bark beetle in the family Curculionidae. It is found in North America.", "target": "species of insect", "baseline_candidates": ["taxon"]}

{"wikidata_id": "Q21000782", "label": "Konstantinos Stivachtis", "source": "Konstantinos Stivachtis (Greek: Κωνσταντίνος Στιβαχτής, born 22 May 1980) is a Greek male volleyball player. He is part of the Greece men's national volleyball team. On club level he plays for Olympiacos.", "target": "Greek volleyball player", "baseline_candidates": ["human"]}

{"wikidata_id": "Q56338504", "label": "embassy of the Philippines to the Holy See", "source": "The Embassy of the Philippines to the Holy See is the diplomatic mission of the Republic of the Philippines to the Holy See, the central government of the Roman Catholic Church. Opened in 1957, it is located along Via Paolo VI in the rione of Borgo, part of Municipio I in central Rome along the border between Italy and Vatican City, and across from St. Peter's Square. It is distinct from the larger Embassy of the Philippines in Rome, the Philippines' diplomatic mission to Italy.", "target": "diplomatic mission of the Philippines to the Holy See", "baseline_candidates": ["embassy"]}

{"wikidata_id": "Q6899359", "label": "Moneysupermarket.com", "source": "Moneysupermarket.com Group PLC is a British price comparison website-based business specialising in financial services. The website enables consumers to compare prices on a range of products, including energy car insurance, home insurance, travel insurance, mortgages, credit cards and loans. The company's 2016 advert was one of the four that received the most complaints from the public in that year. Moneysupermarket is listed on the London Stock Exchange and is a constituent of the FTSE 250 Index.", "target": "British price comparison website-based business", "baseline_candidates": ["public limited company"]}

```

## Phrase 2. Candidate ranking
Similar to Phase 1, there are 2 groups of datasets by 2 ways of data splitting, different topic splitting and random topic splitting. The data distribution is training set ~ 75% (6000 samples), validation set ~ 12.5% (1000 samples), and test set ~ 12.5% (1000 samples). Please get the data here: https://github.com/declare-lab/WikiDes/tree/main/dataset/phase2.

### A detail sample of Phase 2

```
{"source": "Knuthenborg Safaripark is a safari park on the island of Lolland in the southeast of Denmark. It is located 7 km (on Rte 289) to the north of Maribo, near Bandholm. It is one of Lolland's major tourist attractions with over 250,000 visitors annually, and is the largest safari park in northern Europe. It is also the largest natural playground for both children and adults in Denmark. Among others, it houses an arboretum, aviaries, a drive-through safari park, a monkey forest (with baboons, tamarins and lemurs) and a tiger enclosure. Knuthenborg covers a total of 660 hectares (1,600 acres), including the 400-hectare (990-acre) Safaripark. The park is viewable on Google Street View.", "candidate": ["park in Lolland, Denmark", "safari park"], "target": "Safari park in Denmark"}
```

* **source**: the first paragraph of a Wikipedia article.
* **candidate**: the list of machine-generated descriptions from Phase I by beam search.
* **target**:  the **gold description** which we take the first candidate of **baseline_candidates**.

### Some samples of Phase 2:
```
{"source": "Knuthenborg Safaripark is a safari park on the island of Lolland in the southeast of Denmark. It is located 7 km (on Rte 289) to the north of Maribo, near Bandholm. It is one of Lolland's major tourist attractions with over 250,000 visitors annually, and is the largest safari park in northern Europe. It is also the largest natural playground for both children and adults in Denmark. Among others, it houses an arboretum, aviaries, a drive-through safari park, a monkey forest (with baboons, tamarins and lemurs) and a tiger enclosure. Knuthenborg covers a total of 660 hectares (1,600 acres), including the 400-hectare (990-acre) Safaripark. The park is viewable on Google Street View.", "candidate": ["park in Lolland, Denmark", "safari park"], "target": "Safari park in Denmark"}

{"source": "March 2015 was the third month of that common year. The month, which began on a Sunday, ended on a Tuesday after 31 days.", "candidate": ["third month of that common year", "calendar month of a given year", "month starting on Sunday", "March"], "target": "month of 2015"}

{"source": "The Tower of Sacru (Corsican: Torra di Sacru) is a ruined Genoese tower located in the commune of Brando, Haute-Corse on the east coast of the Corsica. Only part of the base survives. The tower was one of a series of coastal defences constructed by the Republic of Genoa between 1530 and 1620 to stem the attacks by Barbary pirates.", "candidate": ["ruined Genoese tower in Brando, Haute-Corse, Italy", "Genoese towers in Corsica"], "target": "genoese coastal defence tower in Corsica"}

{"source": "The Comoro Islands or Comoros (Shikomori Komori; Arabic: جزر القمر, Juzur al-qamar; French: Les Comores) form an archipelago of volcanic islands situated off the southeastern coast of Africa, to the east of Mozambique and northwest of Madagascar. The islands are politically divided between the Union of the Comoros, a sovereign country, and Mayotte, an Overseas Department of France.", "candidate": ["archipelago", "archipelago of volcanic islands"], "target": "archipelago in the Indian Ocean"}

```

# Experiments

## Dependencies

* Python >= 3.6
* spaCy 3.4.1

## Collect data

To collect your own data (randomly), use:

```
python collect_data.py
```

The results will be saved in the folder **"dataset/collected_data.json"**. 

## Phase I: Description Generation by Summarizer

To train the generation model, use:

```
python summarizer.py
```

Note to update these code lines in the file **summarizer.py**:

```
tokenizer = AutoTokenizer.from_pretrained('facebook/bart-base', do_lower_case=False)
    
if __name__ == "__main__":

    # training models: facebook/bart-base, t5-small, t5-base, microsoft/ssr-base, google/t5-v1_1-small, google/t5-v1_1-base
    config = Config(model = 'facebook/bart-base', tokenizer = 'facebook/bart-base', batch_size = 8, \
                    encoder_max_length = 256, decoder_max_length = 32, num_train_epochs = 3)

    train_data, val_data = load_data(config.batch_size, config.tokenizer, config.encoder_max_length, \
                                     config.decoder_max_length, train_file = 'dataset/phrase1/random/training_para_256.json', \
                                     val_file = 'dataset/phrase1/random/validation_para_256.json')

    train(config.model, config.tokenizer, train_data, val_data, num_train_epochs = config.num_train_epochs, \
          batch_size = config.batch_size, output_dir='output/' + config.model_name)

```
* Use the pre-trained model for the training, for example **"facebook/bart-base"**.

*We will add the arguments soon.*

## Phase II: Candidate Ranking by Ranker

To train the ranking model, use:

```
python post_eval.py
```

Note to update these code lines in the file **post_eval.py**:

```
if __name__ == '__main__':

    config = Config(model_name = 'bert-base-cased', tokenizer = 'bert-base-cased', batch_size = 2, max_length = 256)
    train_model(config, splitting_type = 'different', num_epochs = 3, use_sim = True, use_rouge = False, \
                training_file = 'dataset/phrase2/generated_training_para_256_diff.json', \
                val_file = 'dataset/phrase2/generated_validation_para_256_diff.json', \
                test_file = 'dataset/phrase2/generated_test_para_256_diff.json')
```

*We will add the arguments soon.*

## Human evaluation

To see the human concensus, run this command:

```
python human_eval.py
```

To annotate data by human, run:
```
python human_annot.py
```

* Check the code files for details.

# Questions or Issues:
Please ask us at https://github.com/declare-lab/WikiDes/issues or contact to tahoangthang@gmail.com.


# Citation

## APA
Ta, H. T., Rahman, A. B. S., Majumder, N., Hussain, A., Najjar, L., Howard, N., ... & Gelbukh, A. (2022). WikiDes: A Wikipedia-based dataset for generating short descriptions from paragraphs. *Information Fusion*.

## BibTeX
```
@article{Ta_2022,	
doi = {10.1016/j.inffus.2022.09.022},	
url = {https://doi.org/10.1016%2Fj.inffus.2022.09.022},	
year = 2022,	
month = {sep},	
publisher = {Elsevier {BV}},	
author = {Hoang Thang Ta and Abu Bakar Siddiqur Rahman and Navonil Majumder and Amir Hussain and Lotfollah Najjar and Newton Howard and Soujanya Poria and Alexander Gelbukh},	
title = {{WikiDes}: A Wikipedia-based dataset for generating short descriptions from paragraphs},	
journal = {Information Fusion}}
```

# Paper links
* https://doi.org/10.1016%2Fj.inffus.2022.09.022
* https://arxiv.org/abs/2209.13101

# Contact
Hoang Thang Ta, tahoangthang@gmail.com


