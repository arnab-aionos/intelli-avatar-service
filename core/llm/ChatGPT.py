'''
Azure OpenAI Integration for ChatGPT LLM

Supports both standard OpenAI and Azure OpenAI.
Uses Azure OpenAI by default (from config).

pip install openai
'''
import os
from typing import Optional

try:
    # Try new OpenAI library (v1.0+)
    from openai import AzureOpenAI, OpenAI
    USE_NEW_API = True
except ImportError:
    # Fallback to old OpenAI library (v0.x)
    import openai
    USE_NEW_API = False


class ChatGPT:
    """
    ChatGPT LLM wrapper supporting both Azure and standard OpenAI.
    
    By default, uses Azure OpenAI from environment configuration.
    """
    
    def __init__(
        self,
        model_path: str = 'gpt-3.5-turbo',
        api_key: Optional[str] = None,
        proxy_url: Optional[str] = None,
        prefix_prompt: str = 'Please respond concisely in less than 50 words.\n\n',
        # Azure OpenAI specific
        use_azure: bool = True,
        azure_endpoint: Optional[str] = None,
        azure_deployment: Optional[str] = None,
        azure_api_version: str = '2024-02-01'
    ):
        """
        Initialize ChatGPT client.
        
        Args:
            model_path: Model name (for standard OpenAI) or deployment name (for Azure)
            api_key: API key (Azure or OpenAI)
            proxy_url: Optional HTTP proxy URL
            prefix_prompt: System prompt prepended to user messages
            use_azure: Whether to use Azure OpenAI (default: True)
            azure_endpoint: Azure OpenAI endpoint URL
            azure_deployment: Azure deployment name
            azure_api_version: Azure API version
        """
        if proxy_url:
            os.environ['https_proxy'] = proxy_url
            os.environ['http_proxy'] = proxy_url
        
        self.model_path = model_path
        self.prefix_prompt = prefix_prompt
        self.use_azure = use_azure
        
        if USE_NEW_API:
            # New OpenAI library (v1.0+)
            if use_azure:
                self.client = AzureOpenAI(
                    api_key=api_key,
                    azure_endpoint=azure_endpoint,
                    api_version=azure_api_version
                )
                self.deployment = azure_deployment or model_path
            else:
                self.client = OpenAI(api_key=api_key)
                self.deployment = model_path
        else:
            # Old OpenAI library (v0.x)
            if use_azure:
                openai.api_type = "azure"
                openai.api_base = azure_endpoint
                openai.api_version = azure_api_version
                openai.api_key = api_key
                self.deployment = azure_deployment or model_path
            else:
                openai.api_key = api_key
                self.deployment = model_path

    def generate(self, message: str) -> str:
        """
        Generate a response to the user message.
        
        Args:
            message: User's input message
        
        Returns:
            str: Generated response
        """
        try:
            if USE_NEW_API:
                # New API (v1.0+)
                response = self.client.chat.completions.create(
                    model=self.deployment,
                    messages=[
                        {"role": "user", "content": self.prefix_prompt + message}
                    ]
                )
                return response.choices[0].message.content
            else:
                # Old API (v0.x)
                if self.use_azure:
                    response = openai.ChatCompletion.create(
                        engine=self.deployment,  # Azure uses "engine"
                        messages=[
                            {"role": "user", "content": self.prefix_prompt + message}
                        ]
                    )
                else:
                    response = openai.ChatCompletion.create(
                        model=self.deployment,  # Standard OpenAI uses "model"
                        messages=[
                            {"role": "user", "content": self.prefix_prompt + message}
                        ]
                    )
                return response['choices'][0]['message']['content']
                
        except Exception as e:
            print(f"ChatGPT error: {e}")
            return "Sorry, your request encountered an error. Please try again."


def create_chatgpt_from_config():
    """
    Create ChatGPT instance from environment configuration.
    
    Returns:
        ChatGPT: Configured ChatGPT instance
    """
    from config import settings
    
    # Use Azure OpenAI if configured
    if settings.azure_openai_key and settings.azure_openai_endpoint:
        return ChatGPT(
            model_path=settings.azure_openai_deployment,
            api_key=settings.azure_openai_key,
            use_azure=True,
            azure_endpoint=settings.azure_openai_endpoint,
            azure_deployment=settings.azure_openai_deployment,
            azure_api_version=settings.azure_openai_api_version,
            prefix_prompt="You are a helpful AI assistant. Respond naturally and concisely.\n\n"
        )
    # Fallback to standard OpenAI
    elif settings.openai_api_key:
        return ChatGPT(
            model_path='gpt-3.5-turbo',
            api_key=settings.openai_api_key,
            use_azure=False,
            prefix_prompt="You are a helpful AI assistant. Respond naturally and concisely.\n\n"
        )
    else:
        raise ValueError("No OpenAI API key configured (Azure or standard)")


if __name__ == '__main__':
    # Test with config
    llm = create_chatgpt_from_config()
    answer = llm.generate("How do I manage stress?")
    print(answer)