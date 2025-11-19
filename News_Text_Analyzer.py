# --- STEP 1: Import libraries --- #

import requests, pandas,pytorch,tensorflow
!pip install transformers

# --- STEP 2: Initialize Hugging Face models —#

sentiment_model = pipeline("sentiment-analysis",model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",revision="714eb0f")

#---Notes:---#

This line creates a sentiment analysis model using HuggingFace’s pipeline API.

It loads a fine-tuned DistilBERT model specifically trained to classify text into: [This is the popular SST-2 (Stanford Sentiment Treebank v2) dataset]
	•	POSITIVE
	•	NEGATIVE

(a) pipeline("sentiment-analysis") ----->This tells HuggingFace to Load a model that can analyze the sentiment of text.The pipeline abstracts all the complexity such as:
	•	Tokenization
	•	Model inference
	•	Decoding

(b) model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",This selects the specific model you want to use.
  
	•	DistilBERT → A smaller, faster version of BERT
	•	base-uncased → Converts text to lowercase internally
	•	fine-tuned on SST-2 → Model was trained to determine positive or negative tone in English sentences

(c)revision="714eb0f" ------> This chooses a specific version/commit of the model from HuggingFace .This is important because,

✔ Ensures version consistency
✔ Avoids changes in model weights in the future
✔ Makes your project reproducible

If HuggingFace updates the model later, your code still uses the exact same model snapshot.

  Example:

1. sentiment_model("The product quality is excellent!")

Ouput : [{"label": "POSITIVE", "score": 0.998}]

2. sentiment_model("The flight was delayed and the service was terrible.")]

output :  [{"label": "NEGATIVE", "score": 0.996}]


#---- Step 3: # Summarization (Use BART for summarization ---#)

summarizer_model = pipeline("summarization",model="facebook/bart-large-cnn”)

#--Notes--#
                            
This line creates a text-summarization model using the HuggingFace pipeline function.

Key components:

(A) pipeline(“summarization”)
Tells HuggingFace to load a model that can summarize long text into a shorter version.

(B) model=“facebook/bart-large-cnn”
Loads the BART-Large-CNN model from Facebook AI — one of the best models for abstract summarization.

(C) What does BART do?

	•	Reads long paragraphs of news
	•	Understands context
	•	Generates human-like summaries


#---- step 4:# Named Entity Recognition model (BERT fine-tuned for NER) -------- #

ner_model = pipeline("ner",model="dslim/bert-base-NER",grouped_entities=True)

#---Notes---#:

This line creates a model that can detect entities in text.

Key components:

(a) pipeline(“ner”)
Loads a model for Named Entity Recognition.

(b)model=“dslim/bert-base-NER”
Uses a fine-tuned BERT model that detects:
•	PERSON
	•	LOCATION
	•	ORGANIZATION
	•	DATE
	•	PRODUCT
	•	EVENT

(c)grouped_entities=True

Groups multi-word entities together.

Example:
"New", "York" are separate
With this:
"New York" becomes one entity

(d) What does NER do?

It extracts structured information:
	•	People mentioned
	•	Companies
	•	Places
	•	Dates
	•	Key topics



#----- Step 5: Fetch Live News ------------#

API_KEY = “Enter your News API key”

query = "artificial intelligence" [Enter custom keyword to fetch the content news]

url = f"https://newsapi.org/v2/everything?q={query}&language=en&pageSize=5&apiKey={API_KEY}"  

#--- Notes: [https://newsapi.org/v2/everything (News API Endpoint),q={query} Search keyword,language=en Return only English articles,pageSize=5 Limit results to 5 articles,apiKey={API_KEY},Your secret API key (required for authentication) ]


#---------- STEP 6: Calls News API (HTTP GET Request0  & JSON data conversion to python dictionary ------------- #


response = requests.get(url)
data = response.json()

(a) response = requests.get(url)

This line sends an HTTP GET request to the API (NewsAPI) using the URL built earlier in step 5:

The result is stored in response, which is a Response object containing:

	•	status code (200, 404, 500, etc.)
	•	headers
	•	the actual data sent by the API (in JSON format)

(b) data = response.json()

This takes the raw JSON response and converts it into a Python dictionary.

NewsAPI sends data like:

{
  "status": "ok",
  "totalResults": 5,
  "articles": [
    {...},
    {...}
  ]
}

# ----------------- Step 7 :----------------------#

articles = data.get("articles", [])
df = pd.DataFrame(articles)[["title", "description", "url"]]
df.head()


1.articles = data.get("articles", [])

data is the dictionary you received from the NewsAPI response.

.get("articles", []) tries to extract the value under the key "articles"

 -> If "articles" exists → it returns the list of articles.
 -> If "articles" doesn’t exist → it returns an empty list [] instead of crashing. 


2.df = pd.DataFrame(articles)[["title", "description", "url"]]

Converts the articles list into a pandas DataFrame.
Then selects only three columns:
	•	"title"
	•	"description"
	•	"url"

3.df.head()
Shows the first 5 rows of the DataFrame.



















