import os, json

class LLMProvider(object):
    def __init__(self):
        pass

'''
openai
tenacity
tiktoken

num2words
plotly

azure-identity
python-dotenv
langchain
unstructured[all-docs]
faiss-cpu
'''
class GptProvider(LLMProvider):
    def __init__(self, temperature, key, url, version, model_name):
        self.temperature = temperature
        self.chat_props = {
            'AZURE_OPENAI_KEY' : key,
            'AZURE_OPENAI_URL' : url,
            'AZURE_OPENAI_VERSION' : version,
            'AZURE_OPENAI_MODEL' : model_name
        }
        self.client = self.create_gpt_client(self.chat_props)

    def create_gpt_client(self, props):
        #import openai
        from openai import AzureOpenAI
        openai_client = AzureOpenAI(
            api_key = props['AZURE_OPENAI_KEY'],
            api_version = props['AZURE_OPENAI_VERSION'],
            azure_endpoint = props['AZURE_OPENAI_URL']
            )
        return openai_client

    def prompt(self, prompt):
        user_message = [{"role": "user", "content": prompt}]

        response = self.client.chat.completions.create(
            model = self.chat_props['AZURE_OPENAI_MODEL'],
            temperature = self.temperature,
            messages = user_message
        )   
        return response.choices[0].message.content

    def get_embeddings(self, chunk):
        from tenacity import retry, stop_after_attempt, wait_random_exponential

        def before_retry_sleep(retry_state):
            print(f'Rate limited on the OpenAI API, sleeping before retrying...')

        @retry(wait=wait_random_exponential(min=1, max=30), stop=stop_after_attempt(2), before_sleep=before_retry_sleep)
        def generate_embeddings(text, model):
            return self.client.embeddings.create(input=[text], model=model).data[0].embedding

        return generate_embeddings(chunk, self.chat_props['AZURE_OPENAI_MODEL'])

'''
!pip install boto3 aws-bedrock-token-generator  matplotlib==3.5.3 tensorflow-gpu==2.10.0 scikit-learn==1.0.2
'''
class ClaudeProvider(LLMProvider):
    def __init__(self, temperature, region, model_name, max_tokens, anthropic_version):
        self.region = region
        self.model_id = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.anthropic_version = anthropic_version
        self.refresh()

    def refresh(self):
        self.token = self.get_token(self.region)
        os.environ['AWS_BEARER_TOKEN_BEDROCK'] = self.token
        self.client = self.init_boto(self.region)
    
    def init_boto(self, region=None):
        import boto3
        if region is None: return boto3.Session()
        else: return boto3.client(service_name='bedrock-runtime', region_name=region)
        
    def get_token(self, region):
        from aws_bedrock_token_generator import BedrockTokenGenerator
        credentials = self.init_boto().get_credentials()
        return BedrockTokenGenerator().get_token(credentials, region)

    def prompt(self, prompt):
        request = {
          'anthropic_version' : self.anthropic_version,
          'max_tokens' : self.max_tokens,
          'temperature' : self.temperature,
          'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': prompt}]}]
        }
        #response = client.invoke_model_with_response_stream(body=request, modelId=model_id, accept=accept, contentType=contentType)
        #for event in response['body']: print(event['chunk']) 
        request = json.dumps(request)
        response = self.client.invoke_model(modelId=self.model_id, body=request)
        model_response = json.loads(response['body'].read())
        return model_response['content'][0]['text']


def main():
    '''
     4. Return the answer as class of number 1 or 0 in schema <text_class></text_class>.

    Here is a social media post in a given context. Please analyze it and classify whether it contains hate speech.
    Return only the text class=1 if the context is hate speech and class=0 if it is not.
    '''

    instruction = '''Your job is to answer the user's question from the provided context. Follow the following instructions:
                    1. Don't use anything outside of the context and only use provided context.
                    2. Think before you answer and provided a justification as to why you think it's the answer.
                    3. If there are multiple answers to the question, list all the answers down.
                    4. Provide the reference page number where you found the answer.
                    5. When using the context, add also the reference in brackets, for example [info1.pdf]. Do not combine the references, list them separately for example [info1.pdf][info2.pdf].
                    6. If you don't find the answer, return only 'The answer not found.' as an only answer.
                 '''

    instruction = '''Your job is to answer the user's question from the provided context.'''
    
    # we deliberately use incoplete data for evaluation purposes
    user_query = '''Here is the population European countries in csv.
                    Please analyze the population in Europe with respect to the population of each country.'''

    context ='''country,population
                UK,69000000
                Germany,85000000
                Italy,59000000
                Spain,49000000
                France,69000000
                '''

    user_message = f'''
            <instruction>{instruction}</instruction>,
            <user_query>{user_query}</user_query>,
            <context>{context}</context>'''

    props = {}
    props['temperature'] = 0.0

    #props['key'] = 'xx'
    #props['url'] = 'xx'
    #props['version'] = '2025-01-01-preview'
    #props['model_name'] = 'gpt-4o'
    #llm = GptProvider(temperature, key, url, version, model_name)
    #vect = llm.get_embeddings(chunk)

    props['region'] = 'xx'    
    props['model_name'] = 'xx'
    props['anthropic_version'] = 'xx'
    props['max_tokens'] = 500
    llm = ClaudeProvider(**props)

    message = llm.prompt(user_message)
    print(message)

if __name__ == '__main__':    
    main()
'''
    props['region'] = 'xx'    
    props['model_name'] = 'xx'
    props['anthropic_version'] = 'xx'
    props['max_tokens'] = 500    
'''