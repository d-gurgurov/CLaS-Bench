import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
from functools import partial
import os

HF_TOKEN = os.environ.get("HF_TOKEN")
if HF_TOKEN:
    login(token=HF_TOKEN)
    print("Logged into Hugging Face")
else:
    print("HF_TOKEN not found in environment variables")

# ============================================================================
# Language Steering Configuration
# ============================================================================

LANGUAGES = {
    "English": "en",
    "Arabic": "ar",
    "Tibetan": "bo",
    "Danish": "da",
    "German": "de",
    "Spanish": "es",
    "French": "fr",
    "Hindi": "hi",
    "Italian": "it",
    "Japanese": "ja",
    "Korean": "ko",
    "Maltese": "mt",
    "Dutch": "nl",
    "Norwegian": "no",
    "Polish": "pl",
    "Portuguese": "pt",
    "Russian": "ru",
    "Swedish": "sv",
    "Turkish": "tr",
    "Chinese": "zh",
    "Slovak": "sk",
    "Greek": "el",
    "Kazakh": "kk",
    "Swahili": "sw",
    "Georgian": "ka",
    "Ukrainian": "uk",
    "Persian": "fa",
    "Thai": "th",
    "Indonesian": "id",
    "Vietnamese": "vi",
    "Czech": "cs",
    "Romanian": "ro",
}

# Model configurations
MODELS = {
    "Llama-3.1-8B-Instruct": {
        "id": "meta-llama/Llama-3.1-8B-Instruct",
        "vector_suffix": ".llama",
        "neuron_mask_prefix": "llama",
        "activation_suffix": ".llama",
        "num_layers": 32,
        "layers_attr": "model.layers",
        "mlp_attr": "mlp",
        "gate_proj_attr": "gate_proj",
        "activation": "silu",
    },
    "Aya-Expanse-8B": {
        "id": "CohereForAI/aya-expanse-8b",
        "vector_suffix": ".aya",
        "neuron_mask_prefix": "aya",
        "activation_suffix": ".aya",
        "num_layers": 32,
        "layers_attr": "model.layers",
        "mlp_attr": "mlp",
        "gate_proj_attr": "gate_proj",
        "activation": "silu",
    },
}

# THE DIRECTORIES BELOW HAVE TO BE CHANGED TO _aya TO ACCESS AYA-EXPANSE FILES
STEERING_DIR = "CLaS-Bench/identification/data_llama"  # Directory with steering vector files
NEURONS_DIR = "CLaS-Bench/identification/data_llama"  # Directory with neuron mask files (e.g., llama-1, llama-2)
ACTIVATIONS_DIR = "CLaS-Bench/identification/data_llama"  # Directory with activation files (e.g., activation.en.llama)

# ============================================================================
# Global State
# ============================================================================

model = None
tokenizer = None
current_model_name = None
steering_vectors = {}
neuron_masks = {}  # Cache for neuron masks
activation_data = {}  # Cache for activation data
active_hooks = []


def load_model(model_choice):
    """Load the selected model and tokenizer"""
    global model, tokenizer, current_model_name, steering_vectors, neuron_masks, activation_data
    
    model_config = MODELS.get(model_choice)
    if model_config is None:
        return f"⚠️ Unknown model: {model_choice}"
    
    # Check if already loaded
    if current_model_name == model_choice and model is not None:
        return f"✓ {model_choice} already loaded!"
    
    # Clear previous model if switching
    if model is not None:
        print(f"Unloading {current_model_name}...")
        del model
        del tokenizer
        torch.cuda.empty_cache()
        steering_vectors = {}
        neuron_masks = {}
        activation_data = {}
    
    print(f"Loading {model_choice}...")
    model_id = model_config["id"]
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    current_model_name = model_choice
    
    return f"✓ {model_choice} loaded successfully!"


def load_steering_vector(lang_code, model_name):
    """Load a single steering vector for the specified model"""
    global steering_vectors
    
    model_config = MODELS.get(model_name)
    if model_config is None:
        return None
    
    cache_key = f"{model_name}_{lang_code}"
    if cache_key in steering_vectors:
        return steering_vectors[cache_key]
    
    suffix = model_config["vector_suffix"]
    filepath = os.path.join(STEERING_DIR, f"vector.{lang_code}{suffix}")
    
    if not os.path.exists(filepath):
        return None
    
    data = torch.load(filepath, map_location='cpu')
    steering_vectors[cache_key] = data['steering_vectors']
    return steering_vectors[cache_key]


def load_neuron_mask(model_name, k_percent):
    """
    Load neuron mask for the specified model and K%.
    Files are named like: llama-1, llama-2, aya-1, aya-2, etc.
    """
    global neuron_masks
    
    model_config = MODELS.get(model_name)
    if model_config is None:
        return None
    
    cache_key = f"mask_{model_name}_{k_percent}"
    if cache_key in neuron_masks:
        return neuron_masks[cache_key]
    
    prefix = model_config["neuron_mask_prefix"]
    filepath = os.path.join(NEURONS_DIR, f"{prefix}-{k_percent}")
    
    if not os.path.exists(filepath):
        print(f"Neuron mask file not found: {filepath}")
        return None
    
    mask_data = torch.load(filepath, map_location='cpu')
    neuron_masks[cache_key] = mask_data
    return mask_data


def load_activation_data(lang_code, model_name):
    """
    Load activation data for the specified language and model.
    Files are named like: activation.en.llama, activation.ru.aya, etc.
    """
    global activation_data
    
    model_config = MODELS.get(model_name)
    if model_config is None:
        return None
    
    cache_key = f"activation_{model_name}_{lang_code}"
    if cache_key in activation_data:
        return activation_data[cache_key]
    
    suffix = model_config["activation_suffix"]
    filepath = os.path.join(ACTIVATIONS_DIR, f"activation.{lang_code}{suffix}")
    
    if not os.path.exists(filepath):
        print(f"Activation file not found: {filepath}")
        return None
    
    data = torch.load(filepath, map_location='cpu')
    activation_data[cache_key] = data
    return data


def get_model_layers(model, model_name):
    """Get the layers from the model based on model architecture"""
    model_config = MODELS.get(model_name)
    if model_config is None:
        return None
    
    layers_attr = model_config["layers_attr"]
    obj = model
    for attr in layers_attr.split("."):
        obj = getattr(obj, attr)
    return obj


def get_mlp_module(layer, model_name):
    """Get the MLP module from a layer"""
    model_config = MODELS.get(model_name)
    mlp_attr = model_config.get("mlp_attr", "mlp")
    return getattr(layer, mlp_attr)


# ============================================================================
# Diffmean Steering
# ============================================================================

def create_steering_hook(steering_diff, strength):
    """Create a hook function for diffmean steering on hidden states"""
    def hook(module, input, output, layer_idx, steering_diff, strength):
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output
        
        layer_steering = steering_diff[layer_idx].to(hidden_states.device).to(hidden_states.dtype)
        layer_steering = layer_steering / (layer_steering.norm(p=2) + 1e-8)
        
        if hidden_states.dim() == 3:
            hidden_states = hidden_states + strength * layer_steering.unsqueeze(0).unsqueeze(0)
        else:
            hidden_states = hidden_states + strength * layer_steering.unsqueeze(0)
        
        if isinstance(output, tuple):
            return (hidden_states,) + output[1:]
        else:
            return hidden_states
    
    return hook


def apply_diffmean_steering(source_code, target_code, strength, layer_start, layer_end):
    """Apply diffmean steering vectors to the model"""
    global model, active_hooks, current_model_name
    
    remove_steering()
    
    source_vec = load_steering_vector(source_code, current_model_name)
    target_vec = load_steering_vector(target_code, current_model_name)
    
    if source_vec is None:
        return False, f"Steering vector not found for {source_code} ({current_model_name})"
    if target_vec is None:
        return False, f"Steering vector not found for {target_code} ({current_model_name})"
    
    steering_diff = target_vec - source_vec
    
    layers = get_model_layers(model, current_model_name)
    start = int(layer_start)
    end = int(layer_end) + 1
    
    for layer_idx in range(start, min(end, len(layers))):
        layer = layers[layer_idx]
        hook_fn = partial(
            create_steering_hook(steering_diff, strength),
            layer_idx=layer_idx,
            steering_diff=steering_diff,
            strength=strength
        )
        handle = layer.register_forward_hook(hook_fn)
        active_hooks.append(handle)
    
    return True, f"✓ Diffmean: {source_code} → {target_code} (strength={strength}, layers={start}-{end-1})"


# ============================================================================
# Neuron-based Steering
# ============================================================================

def create_neuron_gate_hook(
    layer_idx,
    activate_indices=None,
    activate_values=None,
    deactivate_indices=None,
    activation_strength=1.0,
    deactivation_strength=0.0,
):
    """
    Create a hook for the gate projection to manipulate neuron activations.
    Uses additive intervention by default.
    """
    def hook(module, input, output):
        modified_output = output.clone()
        
        # Deactivate source language neurons
        if deactivate_indices is not None and len(deactivate_indices) > 0:
            deact_idx = deactivate_indices.to(modified_output.device)
            if modified_output.dim() == 3:
                modified_output[:, :, deact_idx] = deactivation_strength
            else:
                modified_output[:, deact_idx] = deactivation_strength
        
        # Activate/boost target language neurons (additive intervention)
        if activate_indices is not None and len(activate_indices) > 0 and activate_values is not None:
            act_idx = activate_indices.to(modified_output.device)
            boost = activate_values.to(modified_output.device).to(modified_output.dtype)
            
            if modified_output.dim() == 3:
                modified_output[:, :, act_idx] = modified_output[:, :, act_idx] + activation_strength * boost.unsqueeze(0).unsqueeze(0)
            else:
                modified_output[:, act_idx] = modified_output[:, act_idx] + activation_strength * boost.unsqueeze(0)
        
        return modified_output
    
    return hook


def apply_neuron_steering(
    source_code, 
    target_code, 
    k_percent, 
    activation_strength, 
    deactivation_strength,
    deactivate_source=True
):
    """
    Apply neuron-based steering to the model across ALL layers.
    
    Args:
        source_code: Source language code
        target_code: Target language code  
        k_percent: Neuron percentage (1, 2, 3, 4, or 5)
        activation_strength: Strength for activating target neurons
        deactivation_strength: Value to set deactivated neurons to
        deactivate_source: Whether to deactivate source language neurons
    """
    global model, active_hooks, current_model_name
    
    remove_steering()
    
    model_config = MODELS.get(current_model_name)
    if model_config is None:
        return False, f"Unknown model: {current_model_name}"
    
    num_layers = model_config["num_layers"]
    
    # Load neuron mask for this K%
    neuron_mask = load_neuron_mask(current_model_name, k_percent)
    if neuron_mask is None:
        return False, f"Neuron mask not found for K={k_percent}% ({current_model_name})"
    
    # Load activation data for target language
    target_activation = load_activation_data(target_code, current_model_name)
    if target_activation is None:
        return False, f"Activation data not found for {target_code} ({current_model_name})"
    
    # Load activation data for source language if deactivating
    source_activation = None
    if deactivate_source and source_code != target_code:
        source_activation = load_activation_data(source_code, current_model_name)
        if source_activation is None:
            print(f"Warning: Activation data not found for source {source_code}, skipping deactivation")
    
    # Get language indices from the mask structure
    # Expected mask structure: list of tensors per language, indexed by language
    lang_names = [
        "en", "ar", "bo", "da", "de", "es", "fr", "hi", "it", "ja", "ko", "mt",
        "nl", "no", "pl", "pt", "ru", "sv", "tr", "zh", "sk", "el", "kk",
        "sw", "ka", "uk", "fa", "th", "id", "vi", "cs", "ro"
    ]
    lang_to_idx = {lang: idx for idx, lang in enumerate(lang_names)}
    
    target_idx = lang_to_idx.get(target_code)
    source_idx = lang_to_idx.get(source_code)
    
    if target_idx is None:
        return False, f"Unknown target language code: {target_code}"
    
    # Get average activations from the loaded data
    target_avg_activations = target_activation.get('average_activations', None)
    source_avg_activations = source_activation.get('average_activations', None) if source_activation else None
    
    # Apply hooks to ALL layers
    layers = get_model_layers(model, current_model_name)
    
    for layer_idx in range(num_layers):
        layer = layers[layer_idx]
        mlp = get_mlp_module(layer, current_model_name)
        
        gate_proj_attr = model_config.get("gate_proj_attr", "gate_proj")
        if not hasattr(mlp, gate_proj_attr):
            print(f"Warning: Could not find gate_proj at layer {layer_idx}")
            continue
        gate_proj = getattr(mlp, gate_proj_attr)
        
        # Get neuron indices for target language at this layer
        # Mask structure: neuron_mask[lang_idx][layer_idx] = tensor of neuron indices
        if isinstance(neuron_mask, list) and target_idx < len(neuron_mask):
            target_layer_mask = neuron_mask[target_idx]
            if isinstance(target_layer_mask, list) and layer_idx < len(target_layer_mask):
                activate_indices = target_layer_mask[layer_idx]
            elif isinstance(target_layer_mask, torch.Tensor):
                activate_indices = target_layer_mask
            else:
                activate_indices = torch.tensor([])
        else:
            activate_indices = torch.tensor([])
        
        # Get activation values for these neurons
        if target_avg_activations is not None and layer_idx < len(target_avg_activations):
            layer_activations = target_avg_activations[layer_idx]
            if len(activate_indices) > 0:
                activate_values = layer_activations[activate_indices]
            else:
                activate_values = torch.tensor([])
        else:
            activate_values = torch.ones(len(activate_indices)) if len(activate_indices) > 0 else torch.tensor([])
        
        # Get deactivation indices for source language
        deactivate_indices = None
        if deactivate_source and source_idx is not None and source_code != target_code:
            if isinstance(neuron_mask, list) and source_idx < len(neuron_mask):
                source_layer_mask = neuron_mask[source_idx]
                if isinstance(source_layer_mask, list) and layer_idx < len(source_layer_mask):
                    deactivate_indices = source_layer_mask[layer_idx]
                elif isinstance(source_layer_mask, torch.Tensor):
                    deactivate_indices = source_layer_mask
        
        # Skip if no neurons to modify at this layer
        if len(activate_indices) == 0 and (deactivate_indices is None or len(deactivate_indices) == 0):
            continue
        
        # Create and register hook
        hook_fn = create_neuron_gate_hook(
            layer_idx=layer_idx,
            activate_indices=activate_indices,
            activate_values=activate_values,
            deactivate_indices=deactivate_indices,
            activation_strength=activation_strength,
            deactivation_strength=deactivation_strength,
        )
        
        handle = gate_proj.register_forward_hook(hook_fn)
        active_hooks.append(handle)
    
    mode = "activate+deactivate" if deactivate_source and source_code != target_code else "activate only"
    return True, f"✓ Neurons ({mode}): {source_code} → {target_code} (K={k_percent}%, act={activation_strength}, deact={deactivation_strength})"


# ============================================================================
# Common Functions
# ============================================================================

def remove_steering():
    """Remove all steering hooks"""
    global active_hooks
    for handle in active_hooks:
        handle.remove()
    active_hooks = []


def generate_response(
    prompt, 
    model_choice, 
    source_lang, 
    target_lang, 
    steering_method,
    # Diffmean parameters
    diffmean_strength,
    layer_start,
    layer_end,
    # Neuron parameters
    neuron_k_percent,
    neuron_activation_strength,
    neuron_deactivation_strength,
    neuron_deactivate_source,
    # Common parameters
    max_tokens, 
):
    """Generate a response with optional steering"""
    global model, tokenizer, current_model_name
    
    if model is None:
        return "⚠️ Please load a model first!"
    
    if current_model_name != model_choice:
        return f"⚠️ Please load {model_choice} first! Currently loaded: {current_model_name}"
    
    source_code = LANGUAGES.get(source_lang)
    target_code = LANGUAGES.get(target_lang)
    
    # Apply steering based on method
    status_msg = ""
    if source_code != target_code:
        if steering_method == "diffmean":
            success, status_msg = apply_diffmean_steering(
                source_code, target_code, diffmean_strength, layer_start, layer_end
            )
        elif steering_method == "neurons":
            success, status_msg = apply_neuron_steering(
                source_code=source_code,
                target_code=target_code,
                k_percent=int(neuron_k_percent),
                activation_strength=neuron_activation_strength,
                deactivation_strength=neuron_deactivation_strength,
                deactivate_source=neuron_deactivate_source
            )
        else:
            return f"⚠️ Unknown steering method: {steering_method}"
        
        if not success:
            return f"⚠️ {status_msg}"
    
    # Format as chat
    messages = [{"role": "user", "content": prompt}]
    
    input_text = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=int(max_tokens),
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    
    remove_steering()
    
    if status_msg:
        return f"{status_msg}\n\n---\n\n{response}"
    return response


# ============================================================================
# Gradio Interface
# ============================================================================

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono&display=swap');

* {
    font-family: 'IBM Plex Sans', sans-serif !important;
}

.gradio-container {
    max-width: 900px !important;
    margin: auto !important;
    background: #ffffff !important;
    min-height: 100vh;
}

.main-header {
    text-align: center;
    padding: 2rem 0;
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.5rem;
    font-weight: 600;
    letter-spacing: -0.02em;
}

.sub-header {
    text-align: center;
    color: #666;
    margin-bottom: 2rem;
    font-size: 1rem;
}

.control-panel {
    background: #f8f9fa !important;
    border: 1px solid #e9ecef !important;
    border-radius: 16px !important;
    padding: 1.5rem !important;
}

.arrow-indicator {
    font-size: 2rem;
    color: #667eea;
    text-align: center;
    padding: 1rem;
}

textarea, input {
    background: #ffffff !important;
    border: 1px solid #dee2e6 !important;
    border-radius: 8px !important;
    color: #212529 !important;
    font-family: 'IBM Plex Mono', monospace !important;
}

textarea:focus, input:focus {
    border-color: #667eea !important;
    box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2) !important;
}

button.primary {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    transition: transform 0.2s, box-shadow 0.2s !important;
    color: #000000 !important;
}

button.primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3) !important;
    color: #000000 !important;
}

button.primary span {
    color: #000000 !important;
}

.output-box {
    background: #f8f9fa !important;
    border: 1px solid #e9ecef !important;
    border-radius: 12px !important;
    min-height: 200px !important;
}

label {
    color: #495057 !important;
    font-weight: 500 !important;
}

.slider input[type="range"] {
    accent-color: #667eea !important;
}

/* Method selection styling */
.method-selector {
    background: #f8f9fa !important;
    border-radius: 12px !important;
    padding: 1rem !important;
    margin: 1rem 0 !important;
}

/* Active method panel highlight */
.method-panel-active {
    background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%) !important;
    border: 2px solid #667eea !important;
    border-radius: 12px !important;
    padding: 1rem !important;
    margin-top: 0.5rem !important;
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.15) !important;
}

/* Inactive method panel */
.method-panel-inactive {
    background: #f5f5f5 !important;
    border: 1px dashed #ccc !important;
    border-radius: 12px !important;
    padding: 1rem !important;
    margin-top: 0.5rem !important;
    opacity: 0.6 !important;
}

/* Method radio buttons styling */
.method-selector label {
    font-weight: 600 !important;
}

.method-selector input[type="radio"]:checked + label {
    color: #667eea !important;
}
"""

def create_interface():
    with gr.Blocks(css=CUSTOM_CSS, theme=gr.themes.Base(primary_hue="purple")) as demo:
        
        gr.HTML("""
            <h1 class="main-header">🧭 Language Steering</h1>
            <p class="sub-header">Steer LLM responses between languages using activation engineering</p>
        """)
        
        # Model selection and loading section
        with gr.Row():
            model_choice = gr.Dropdown(
                choices=list(MODELS.keys()),
                value="Llama-3.1-8B-Instruct",
                label="🤖 Select Model",
                info="Choose the model to use",
                scale=2
            )
            load_btn = gr.Button("🚀 Load Model", variant="primary", scale=1)
            load_status = gr.Textbox(label="Status", interactive=False, scale=3)
        
        gr.HTML("<hr style='border-color: rgba(0,0,0,0.1); margin: 1.5rem 0;'>")
        
        # Language selection
        with gr.Row(elem_classes="control-panel"):
            with gr.Column(scale=2):
                source_lang = gr.Dropdown(
                    choices=list(LANGUAGES.keys()),
                    value="English",
                    label="📤 Source Language",
                    info="Language of your prompt"
                )
            
            gr.HTML('<div class="arrow-indicator">→</div>')
            
            with gr.Column(scale=2):
                target_lang = gr.Dropdown(
                    choices=list(LANGUAGES.keys()),
                    value="French", 
                    label="📥 Target Language",
                    info="Desired response language"
                )
        
        # Steering method selection with clear visual indicator
        gr.HTML("<h3 style='margin-top: 1.5rem; color: #333;'>🔧 Steering Method</h3>")
        
        with gr.Row(elem_classes="method-selector"):
            steering_method = gr.Radio(
                choices=["diffmean", "neurons"],
                value="diffmean",
                label="Select Method",
                info="Diffmean: hidden state steering | Neurons: MLP gate manipulation",
                interactive=True
            )
        
        # Method indicator
        method_indicator = gr.HTML(
            value='<div style="text-align: center; padding: 0.5rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 8px; font-weight: 600;">📊 Active: Diffmean Steering</div>'
        )
        
        # Diffmean parameters panel
        with gr.Group(visible=True, elem_classes="method-panel-active") as diffmean_panel:
            gr.Markdown("### ⚡ Diffmean Parameters")
            diffmean_strength = gr.Slider(
                minimum=0.0,
                maximum=10.0,
                value=1.0,
                step=0.1,
                label="Steering Strength",
                info="Higher = stronger language shift"
            )
            with gr.Row():
                layer_start = gr.Slider(
                    minimum=0,
                    maximum=31,
                    value=0,
                    step=1,
                    label="🔽 Start Layer",
                    info="First layer to apply steering"
                )
                layer_end = gr.Slider(
                    minimum=0,
                    maximum=31,
                    value=31,
                    step=1,
                    label="🔼 End Layer",
                    info="Last layer to apply steering"
                )
        
        # Neuron parameters panel
        with gr.Group(visible=False, elem_classes="method-panel-inactive") as neuron_panel:
            gr.Markdown("### 🧠 Neuron Parameters")
            gr.Markdown("*Manipulation applied across ALL layers*")
            neuron_k_percent = gr.Radio(
                choices=[1, 2, 3, 4, 5],
                value=1,
                label="📊 Neuron Percentage (K%)",
                info="Top K% of language-specific neurons to manipulate"
            )
            with gr.Row():
                neuron_activation_strength = gr.Slider(
                    minimum=0.0,
                    maximum=10.0,
                    value=1.0,
                    step=0.1,
                    label="⬆️ Activation Strength",
                    info="Strength for boosting target neurons (additive)"
                )
                neuron_deactivation_strength = gr.Slider(
                    minimum=-5.0,
                    maximum=5.0,
                    value=0.0,
                    step=0.1,
                    label="⬇️ Deactivation Value",
                    info="Value to set deactivated neurons to"
                )
            neuron_deactivate_source = gr.Checkbox(
                value=True,
                label="🚫 Deactivate Source Neurons",
                info="Also deactivate source language neurons"
            )
        
        # Common controls
        with gr.Row():
            max_tokens = gr.Slider(
                minimum=64,
                maximum=512,
                value=256,
                step=32,
                label="📏 Max Tokens",
                info="Maximum response length"
            )
        
        # Input/Output
        prompt = gr.Textbox(
            label="💬 Your Prompt",
            placeholder="Enter your message here...",
            lines=3
        )
        
        generate_btn = gr.Button("✨ Generate", variant="primary", size="lg")
        
        output = gr.Textbox(
            label="🤖 Response",
            lines=10,
            elem_classes="output-box"
        )
        
        # Toggle visibility and styling of parameter panels based on method selection
        def toggle_method_panels(method):
            if method == "diffmean":
                indicator_html = '<div style="text-align: center; padding: 0.5rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 8px; font-weight: 600;">📊 Active: Diffmean Steering (Hidden States)</div>'
                return (
                    gr.update(visible=True, elem_classes="method-panel-active"),
                    gr.update(visible=False, elem_classes="method-panel-inactive"),
                    indicator_html
                )
            else:
                indicator_html = '<div style="text-align: center; padding: 0.5rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 8px; font-weight: 600;">🧠 Active: Neuron Steering (MLP Gates)</div>'
                return (
                    gr.update(visible=False, elem_classes="method-panel-inactive"),
                    gr.update(visible=True, elem_classes="method-panel-active"),
                    indicator_html
                )
        
        steering_method.change(
            fn=toggle_method_panels,
            inputs=[steering_method],
            outputs=[diffmean_panel, neuron_panel, method_indicator]
        )
        
        # Example prompts
        gr.Examples(
            examples=[
                ["What is the capital of France?", "Llama-3.1-8B-Instruct", "English", "French", "diffmean", 5.0, 20, 21, 1, 1.0, 0.0, True, 256],
                ["Explain quantum computing in simple terms.", "Llama-3.1-8B-Instruct", "English", "German", "diffmean", 5.0, 20, 21, 1, 1.0, 0.0, True, 256],
                ["What is the capital of France?", "Llama-3.1-8B-Instruct", "English", "French", "neurons", 1.0, 0, 31, 2, 1.0, 0.0, True, 256],
                ["Write a short poem about the ocean.", "Aya-Expanse-8B", "English", "Spanish", "neurons", 1.0, 0, 31, 3, 2.0, 0.0, False, 256],
            ],
            inputs=[
                prompt, model_choice, source_lang, target_lang, steering_method,
                diffmean_strength, layer_start, layer_end,
                neuron_k_percent, neuron_activation_strength,
                neuron_deactivation_strength, neuron_deactivate_source,
                max_tokens
            ],
        )
        
        # Footer
        gr.HTML("""
            <div style="text-align: center; margin-top: 2rem; color: #666; font-size: 0.85rem;">
                <p>Built with 🤗 Transformers & Gradio</p>
                <p>Models: meta-llama/Llama-3.1-8B-Instruct | CohereForAI/aya-expanse-8b</p>
                <p><b>Diffmean:</b> Steering via hidden state differences | <b>Neurons:</b> MLP gate manipulation (additive)</p>
            </div>
        """)
        
        # Event handlers
        load_btn.click(fn=load_model, inputs=[model_choice], outputs=load_status)
        
        generate_btn.click(
            fn=generate_response,
            inputs=[
                prompt, model_choice, source_lang, target_lang, steering_method,
                diffmean_strength, layer_start, layer_end,
                neuron_k_percent, neuron_activation_strength,
                neuron_deactivation_strength, neuron_deactivate_source,
                max_tokens
            ],
            outputs=output
        )
        
        prompt.submit(
            fn=generate_response,
            inputs=[
                prompt, model_choice, source_lang, target_lang, steering_method,
                diffmean_strength, layer_start, layer_end,
                neuron_k_percent, neuron_activation_strength,
                neuron_deactivation_strength, neuron_deactivate_source,
                max_tokens
            ],
            outputs=output
        )
    
    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch()
