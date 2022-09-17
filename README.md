# WikiDes: A Wikipedia-based summarization dataset

We present a novel dataset for generating descriptions of Wikidata from Wikipedia paragraphs. This is a problem of both extreme summarization and indicative summarization. However, after using ranking models, the problem is likely more informative summarization.

# Dataset 

The data contain over 80k samples in the file **collected_data.json**.  There are 2 phases of training, description generation and candidate ranking. 

## Phrase 1. Description generation 
We consider Wikidata instances (https://www.wikidata.org/wiki/Help:Basic_membership_properties#instance_of_(P31)) as topics of samples. The data distribution is training set ~ 80%, validation set ~ 10%, and test set ~ 10%. We use first 256 tokens in Wikipedia first paragraphs as the documents in the training.
* **topic-exclusive split (diff)**: The data is split in different topics. The training set will have different topics from validation and test sets. The distribution of training, validation, and test sets is 65,772/7,820/7,827. Please get the file here: https://github.com/declare-lab/WikiDes/tree/main/dataset/phase1/diff.
* **topic-independent split (random)**: The data is split in random topics. All sets will have random topics. The distribution of training, validation, and test sets is 68,296/8,540/8,542.  Note that we did not filter empty Wikidata instances in this split.  Please get the file here: https://github.com/declare-lab/WikiDes/tree/main/dataset/phase1/random

### A detail sample of Phase 1

```
{"wikidata_id": "Q55135146", "label": "Xyleborus intrusus", "source": "Xyleborus intrusus is a species of typical bark beetle in the family Curculionidae. It is found in North America.", "target": "species of insect", "baseline_candidates": ["taxon"]}
```

* **wikidata_id**: the identifier of a Wikidata item, https://www.wikidata.org/wiki/Wikidata:Identifiers
* **label**: the label of a Wikidata item, usually as the article Wikipedia title, https://www.wikidata.org/wiki/Help:Label
* **source**: the first paragraph of a Wikipedia article
* **target**: the description of a Wikidata item or the **gold description**, https://www.wikidata.org/wiki/Help:Description
* **baseline_candidates**: a list of instances (P31) of a Wiki item, https://www.wikidata.org/wiki/Help:Basic_membership_properties#instance_of_(P31)


### Some samples of Phase 1:
```
{"wikidata_id": "Q55135146", "label": "Xyleborus intrusus", "source": "Xyleborus intrusus is a species of typical bark beetle in the family Curculionidae. It is found in North America.", "target": "species of insect", "baseline_candidates": ["taxon"]}

{"wikidata_id": "Q21000782", "label": "Konstantinos Stivachtis", "source": "Konstantinos Stivachtis (Greek: Κωνσταντίνος Στιβαχτής, born 22 May 1980) is a Greek male volleyball player. He is part of the Greece men's national volleyball team. On club level he plays for Olympiacos.", "target": "Greek volleyball player", "baseline_candidates": ["human"]}

{"wikidata_id": "Q56338504", "label": "embassy of the Philippines to the Holy See", "source": "The Embassy of the Philippines to the Holy See is the diplomatic mission of the Republic of the Philippines to the Holy See, the central government of the Roman Catholic Church. Opened in 1957, it is located along Via Paolo VI in the rione of Borgo, part of Municipio I in central Rome along the border between Italy and Vatican City, and across from St. Peter's Square. It is distinct from the larger Embassy of the Philippines in Rome, the Philippines' diplomatic mission to Italy.", "target": "diplomatic mission of the Philippines to the Holy See", "baseline_candidates": ["embassy"]}

{"wikidata_id": "Q6899359", "label": "Moneysupermarket.com", "source": "Moneysupermarket.com Group PLC is a British price comparison website-based business specialising in financial services. The website enables consumers to compare prices on a range of products, including energy car insurance, home insurance, travel insurance, mortgages, credit cards and loans. The company's 2016 advert was one of the four that received the most complaints from the public in that year. Moneysupermarket is listed on the London Stock Exchange and is a constituent of the FTSE 250 Index.", "target": "British price comparison website-based business", "baseline_candidates": ["public limited company"]}

```

## Phrase 2. Candidate ranking
Similar to Phase 1, there are 2 groups of datasets by 2 ways of data splitting, different topic splitting and random topic splitting. The data distribution is training set ~ 75% (6000 samples), validation set ~ 12.5% (1000 samples), and test set ~ 12.5% (1000 samples). Please get the data here: https://github.com/declare-lab/WikiDes/tree/main/dataset/phase2

### A detail sample of Phase 2

```
{"source": "Knuthenborg Safaripark is a safari park on the island of Lolland in the southeast of Denmark. It is located 7 km (on Rte 289) to the north of Maribo, near Bandholm. It is one of Lolland's major tourist attractions with over 250,000 visitors annually, and is the largest safari park in northern Europe. It is also the largest natural playground for both children and adults in Denmark. Among others, it houses an arboretum, aviaries, a drive-through safari park, a monkey forest (with baboons, tamarins and lemurs) and a tiger enclosure. Knuthenborg covers a total of 660 hectares (1,600 acres), including the 400-hectare (990-acre) Safaripark. The park is viewable on Google Street View.", "candidate": ["park in Lolland, Denmark", "safari park"], "target": "Safari park in Denmark"}
```

* **source**: the first paragraph of a Wikipedia article
* **candidate**: the list of machine-generated descriptions from Phase I by beam search
* **target**:  the **gold description** in Phase I

### Some samples of Phase 2:
```
{"source": "Knuthenborg Safaripark is a safari park on the island of Lolland in the southeast of Denmark. It is located 7 km (on Rte 289) to the north of Maribo, near Bandholm. It is one of Lolland's major tourist attractions with over 250,000 visitors annually, and is the largest safari park in northern Europe. It is also the largest natural playground for both children and adults in Denmark. Among others, it houses an arboretum, aviaries, a drive-through safari park, a monkey forest (with baboons, tamarins and lemurs) and a tiger enclosure. Knuthenborg covers a total of 660 hectares (1,600 acres), including the 400-hectare (990-acre) Safaripark. The park is viewable on Google Street View.", "candidate": ["park in Lolland, Denmark", "safari park"], "target": "Safari park in Denmark"}

{"source": "March 2015 was the third month of that common year. The month, which began on a Sunday, ended on a Tuesday after 31 days.", "candidate": ["third month of that common year", "calendar month of a given year", "month starting on Sunday", "March"], "target": "month of 2015"}

{"source": "The Tower of Sacru (Corsican: Torra di Sacru) is a ruined Genoese tower located in the commune of Brando, Haute-Corse on the east coast of the Corsica. Only part of the base survives. The tower was one of a series of coastal defences constructed by the Republic of Genoa between 1530 and 1620 to stem the attacks by Barbary pirates.", "candidate": ["ruined Genoese tower in Brando, Haute-Corse, Italy", "Genoese towers in Corsica"], "target": "genoese coastal defence tower in Corsica"}

{"source": "The Comoro Islands or Comoros (Shikomori Komori; Arabic: جزر القمر, Juzur al-qamar; French: Les Comores) form an archipelago of volcanic islands situated off the southeastern coast of Africa, to the east of Mozambique and northwest of Madagascar. The islands are politically divided between the Union of the Comoros, a sovereign country, and Mayotte, an Overseas Department of France.", "candidate": ["archipelago", "archipelago of volcanic islands"], "target": "archipelago in the Indian Ocean"}

```

# Experiments
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

*We will add the arguments soon.*

## Phase II: Candidate Ranking by Ranker


# Publication
* "WikiDes: A Wikipedia-Based Dataset for Generating Short Descriptions from Paragraphs"

# Laboratories
* DeCLaRe Lab, https://declare-lab.net/
* CIC IPN, http://nlp.cic.ipn.mx/

# Contact
Hoang Thang Ta, tahoangthang@gmail.com


