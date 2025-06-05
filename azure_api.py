
import logging
import openai

import clemcore.backends as backends
import clemcore.backends.openai_api as openai_api

logger = logging.getLogger(__name__)

NAME = 'azure'


class AzureOpenAI(openai_api.OpenAI):
    '''Backend class for accessing Azure OpenAI remote APIs.'''

    def _make_api_client(self):
        creds = backends.load_credentials(NAME)
        return openai.AzureOpenAI(
            api_version='2025-04-01-preview',
            azure_endpoint=creds[NAME]['base_url'],
            api_key=creds[NAME]['api_key']
        )

