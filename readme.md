# Typeform Extractor

Simple class to make data extraction from Typeform easier. Includes sentiment analysis from Amazon Comprehend (AWS).

***

## The famous three step process:
#### 1. Create a TypeformExtractor object
    credentials = {
        'aws_public_key': 'Your aws public key',
        'aws_private_key': 'Your aws private key',
        'typeform_token': 'Your typeform token'
    }
    extractor = TypeformExtractor(credentials=credentials)

Required parameters:
- **credentials:** AWS and TypeForm credentials to connect to their APIs

Optional parameters:
- **page_size:** How many results to get on each request
- **field_prefix:** Add a prefix to unnamed fields
- **debug:** If True, it will print debug annotations
<br/><br/><br/>

#### 2. Call extract method
    form_id = 'Your_form_id'
    data = test.extract(form_id=form_id)

Required parameters:
- **form_id:** Id of the form to get data from

Optional parameters:
- **field_names:** Dict with all column names to change
- **sentiment:** Columns to analyze with sentiment analysis. If empty, no analysis will be made
- **fixed_fields:** Constant columns to add to the Dataframe, Its value will be the same in all rows
- **auto_translate:** If True, it will fetch column names from Typeform API. Use this with caution, they may not be the names you want or may be too long. It will only fetch columns that do not appear in **field_names**.
<br/><br/>

#### 3. Press play!
<br/><br/>
A more detailed example is located in test.ipynb notebook

## Authors:
- Daniel Vivas - hello@danielvivas.com
- Julia Martínez - jmtcorreo@gmx.es
