import tiktoken

def num_tokens_from_messages(messages, model="gpt-4o-mini"):
    """Returns the number of tokens used by a list of messages for GPT-4 or GPT-4o-mini."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # Fallback to o200k for 4o-mini or cl100k for gpt-4
        encoding = tiktoken.get_encoding("o200k_base") if "4o" in model else tiktoken.get_encoding("cl100k_base")

    # Both GPT-4 and GPT-4o models share these specific message overhead constants
    tokens_per_message = 3
    tokens_per_name = 1
        
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens

def message_process(message, max_token=128000, model="gpt-4o-mini", stopwords_process=None):
    """Truncates message content to fit within the max_token limit using model-specific encoding."""
    
    # 5 token buffer for message formatting overhead
    stop_num = max_token - 5 
    
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("o200k_base") if "4o" in model else tiktoken.get_encoding("cl100k_base")
        
    current_tokens = 4 # Base overhead
    
    for key, value in message.items():
        if stopwords_process: 
            value = stopwords_process(value)
            
        words = value.split(" ")
        content_list = []
        
        for word in words:
            # Estimate word tokens (+1 for the space)
            word_tokens = len(encoding.encode(word)) + 1 
            if current_tokens + word_tokens < stop_num:
                content_list.append(word)
                current_tokens += word_tokens
            else:
                break
                
    return {"role": message["role"], "content": " ".join(content_list)}