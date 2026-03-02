import os
from typing import List, Dict, Any, Optional
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langsmith import traceable
from langsmith.wrappers import wrap_openai
import httpx

load_dotenv()


class GroqChatLLM:
    def __init__(self):
        """Initialize LLM with LangSmith tracing enabled."""
        self.model = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
        self.api_key = os.getenv("GROQ_API_KEY")
        self.temperature = float(os.getenv("TEMPERATURE", "0"))
        self.max_tokens = int(os.getenv("MAX_OUTPUT_TOKENS", "16000"))
        
        # Initialize LLM
        self.llm = ChatGroq(
            model=self.model,
            api_key=self.api_key,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            http_client=httpx.Client(verify=False),
            model_kwargs={
                "seed": int(os.getenv("LLM_SEED", "42")),
                "top_p": float(os.getenv("TOP_P", "0.95")),
            }
        )
    
    @traceable(
        name="groq_chat_completion",
        tags=["llm", "groq", "test-generation"],
        metadata={"model": os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")}
    )
    def chat(
        self, 
        system_prompt: str, 
        user_prompt: str,
        run_name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Execute chat completion with LangSmith tracing.
        
        Args:
            system_prompt: System instruction
            user_prompt: User query
            run_name: Optional name for this specific run
            tags: Optional tags for categorization
            metadata: Optional metadata to attach
        
        Returns:
            Model response content
        """
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        # Add metadata for LangSmith
        extra_metadata = {
            "system_prompt_length": len(system_prompt),
            "user_prompt_length": len(user_prompt),
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        
        if metadata:
            extra_metadata.update(metadata)

        resp = self.llm.invoke(
            messages,
            config={
                "run_name": run_name or "groq_chat",
                "tags": tags or [],
                "metadata": extra_metadata
            }
        )
        
        content = resp.content
        
        return content
    
    @traceable(name="batch_chat_completion", tags=["llm", "groq", "batch"])
    def batch_chat(
        self,
        system_prompt: str,
        user_prompts: List[str],
        run_name: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> List[str]:
        """
        Execute multiple chat completions in batch.
        
        Args:
            system_prompt: System instruction (same for all)
            user_prompts: List of user queries
            run_name: Optional name for this batch
            tags: Optional tags
        
        Returns:
            List of model responses
        """
        responses = []
        
        for idx, user_prompt in enumerate(user_prompts):
            response = self.chat(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                run_name=f"{run_name or 'batch'}_{idx}",
                tags=tags,
                metadata={"batch_index": idx, "batch_size": len(user_prompts)}
            )
            responses.append(response)
        
        return responses