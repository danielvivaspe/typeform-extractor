import boto3
import requests
import pandas as pd
import boto3.exceptions


class TypeformExtractor:
    """
    Simple class to extract data from TypeForm Responses API and analyze text sentiments with Amazon Comprehend (AWS)

    AUTHORS:
        Daniel Vivas
        Julia Martinez Tapia
    """

    credentials = None
    aws_client = None
    last_token = ''
    df = None

    field_prefix = None
    page_size = None
    debug = None

    def __init__(self, credentials: dict, page_size: int = 500, field_prefix: str = 'field_', debug: bool = False):
        """
        :param credentials: AWS and TypeForm credentials to connect to their APIs
        :param page_size: How many results to get on each request
        :param field_prefix: Add a prefix to unnamed fields
        :param debug: If True, it will print debug annotations
        """

        if ('typeform_token' not in credentials.keys()):
            raise Exception('Typeform token is missing')

        self.credentials = credentials

        self.field_prefix = field_prefix
        self.page_size = page_size
        self.debug = debug

        self.aws_client = boto3.client(service_name='comprehend', region_name='us-east-2',
                                       aws_access_key_id=self.credentials['aws_public_key'],
                                       aws_secret_access_key=self.credentials['aws_private_key'])

    def analyze_sentiment(self, *args: str) -> dict:
        """
        Connects to Amazon Comprehend API to analyze sentiment of given texts

        :param args: Texts to analyze
        :return: Percentages of sentiments and sentiment label
        """
        text = ' '.join(args)
        sentiment = None

        try:
            result = self.aws_client.detect_sentiment(Text=text, LanguageCode='es')
            sentiment = result['SentimentScore']
            sentiment['Sentiment'] = result['Sentiment']
        except self.aws_client.exceptions.TextSizeLimitExceededException:
            if self.debug:
                print('Text size limit exceeded for sentiment analysis')

        return sentiment

    def retrieve_data(self, form_id: str) -> dict:
        """
        Fetches form data from Typeform Responses API

        :param form_id: Form ID to get data from
        :return: JSON with fetched raw data
        """

        url = f"https://api.typeform.com/forms/{form_id}/responses"
        params = {
            "page_size": self.page_size,
            'before': self.last_token
        }
        headers = {
            'Authorization': self.credentials['typeform_token']
        }
        data = requests.get(url, headers=headers, params=params)
        return data.json()

    def generate_row(self, data: dict, fixed_fields: dict, sentiment: list):
        """
        Processes raw data to generate rows, gets the analysis sentiment and appends it to the dataframe

        :param data: Raw date fetched from Typeform API
        :param fixed_fields: Fixed columns to add to the dataframe
        :param sentiment: If not empty, it will analyze sentiments based on fields included on this list
        """
        for item in data['items']:

            row = {}

            for key, value in fixed_fields.items():
                row[key] = value

            row['landing_id'] = item['landing_id']
            row['token'] = item['token']
            row['response_id'] = item['response_id']
            row['landed_at'] = item['landed_at']
            row['submitted_at'] = item['submitted_at']
            row['user_agent'] = item['metadata']['user_agent']

            texts = []

            for answer in item['answers']:

                field_id = answer['field']['id']
                field_type = answer['field']['type']

                if field_type == "short_text" or field_type == "long_text":
                    row[self.field_prefix + field_id] = answer['text']
                    if field_type == "long_text" and (
                            (self.field_prefix + field_id in sentiment) or (field_id in sentiment)):
                        texts.append(answer['text'])
                elif field_type == "multiple_choice":
                    if 'choice' in answer.keys():
                        if 'label' in answer['choice'].keys():
                            row[self.field_prefix + field_id] = answer['choice']['label']
                    elif 'choices' in answer.keys():
                        row[self.field_prefix + field_id] = answer['choices']['labels']
                    else:
                        raise Exception('Field type not included in multiple_choice')
                elif field_type == "opinion_scale":
                    row[self.field_prefix + field_id] = answer['number']
                elif field_type == "email":
                    row[self.field_prefix + field_id] = answer['email']
                elif field_type == "yes_no":
                    row[self.field_prefix + field_id] = answer['boolean']
                elif field_type == "number":
                    row[self.field_prefix + field_id] = answer['number']
                elif field_type == "phone_number":
                    row[self.field_prefix + field_id] = answer['phone_number']
                elif field_type == "date":
                    row[self.field_prefix + field_id] = answer['date']
                elif field_type == "picture_choice":
                    row[self.field_prefix + field_id] = answer['choice']['label']
                else:
                    raise Exception('Field type not recognized')

            if len(texts) > 0 and sentiment:
                sentiment_label = self.analyze_sentiment(*texts)

                if sentiment_label is not None:
                    row['Sentiment'] = sentiment_label["Sentiment"]
                else:
                    row['Sentiment'] = 'Not analyzed'

                texts = []

            self.df = self.df.append(row, ignore_index=True)

            # To get the last token for the next requests
            self.last_token = item['token']

    def get_fields(self, data: dict, fixed_columns: dict, sentiment: list) -> list:
        """
        Iterates over raw data and creates a list with fixed columns, submission details and distinct fields the form has

        :param data: Raw data fetched from Typeform API
        :param fixed_columns: Fixed columns to add to the dataframe
        :param sentiment: If not empty, it will analyze sentiments based on fields included on this list
        :return: Columns to be included in the dataframe with no translated names
        """

        columns = []
        fields = set()

        for key in fixed_columns.keys():
            columns.append(key)

        columns.extend(['landing_id', 'token', 'response_id', 'landed_at', 'submitted_at', 'user_agent'])

        for item in data['items']:
            for answer in item['answers']:
                name = self.field_prefix + answer['field']['id']
                fields.add(name)

        columns.extend(fields)

        if len(sentiment) > 0:
            columns.append('Sentiment')

        return columns

    def translate_fields(self, field_names: dict):
        """
        Changes column names found in field_names

        :param field_names: Dict with field names to replace. Keys are the old values and values the new ones.
        """

        fields = self.df.columns

        for field in fields:
            if field in field_names:  # If it appears with prefix
                self.df.rename(columns={field: field_names[field]}, inplace=True)
            elif field.replace(self.field_prefix, '') in field_names:  # If it appears with NO prefix
                self.df.rename(columns={field: field_names[field.replace(self.field_prefix, '')]},
                               inplace=True)

    def extract(self, form_id: str, field_names: dict = None, sentiment: list = [],
                fixed_fields: dict = {}) -> pd.DataFrame:
        """
        Main function. Fetches the data, processes it and returns structured dataframe

        :param form_id: Id of the form to get data from
        :param field_names: Dict with all column names to change
        :param sentiment: Columns to analyze with sentiment analysis. If empty, no analysis will be made
        :param fixed_fields: Constant columns to add to the Dataframe, Its value will be the same in all rows
        :return: Structured dataframe with all translated fields
        """

        # If sentiment analysis is activated, AWS credentials need to be set
        if (len(sentiment) != 0 and ('aws_public_key' not in self.credentials.keys() or self.credentials[
            'aws_public_key'] == '' or 'aws_private_key' not in self.credentials.keys() or self.credentials[
                                         'aws_private_key'] == '')):
            raise Exception('AWS credentials are malformed or missing')

        # Reset dataframe from previous results
        self.df = None

        form_id = str(form_id)

        # ------------------- First request -------------------#
        data = self.retrieve_data(form_id)

        # ---------------- Get distinct columns ---------------#
        columns = self.get_fields(data, fixed_fields, sentiment)

        self.df = pd.DataFrame(columns=columns)

        while len(data['items']) > 0:
            self.generate_row(data, fixed_fields, sentiment)
            data = self.retrieve_data(form_id)

        if field_names is not None:
            self.translate_fields(field_names)

        return self.df
