### Part 1: How to make it pip installable

Create a new folder on your computer to hold your SDK. Arrange the files exactly like this:

Plaintext

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   lifeline-sdk/  ├── setup.py             # Tells pip how to install your package  └── lifeline/            # This is the actual Python package folder      ├── __init__.py      # Empty file (or just imports the client)      └── client.py        # Your code goes here   `

**1\. lifeline/\_\_init\_\_.py**Make it easy for users to import your class by putting this single line inside \_\_init\_\_.py:

Python

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   from .client import LifelineClient   `

**2\. setup.py**Create this file in the root folder and paste this code:

Python

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   from setuptools import setup, find_packages  setup(      name="lifeline-sdk",      version="1.0.0",      description="Official Python SDK for the Lifeline ECG Vision API",      author="Your Name",      packages=find_packages(),      install_requires=[          "requests>=2.25.0", # The only external library your client needs      ],      python_requires=">=3.7",  )   `

**How to install it:**

*   **Locally (for testing):** Open your terminal, navigate to the lifeline-sdk folder, and run: pip install -e .
    
*   **From GitHub:** If you upload this folder to a public GitHub repo, anyone can install it by running: pip install git+https://github.com/yourusername/lifeline-sdk.git
    

### Part 2: Official SDK Documentation

Here is the markdown documentation you can put in your README.md or provide to developers using your system.

Lifeline ECG Vision SDK
=======================

The Lifeline SDK provides a clean, Pythonic interface to interact with the Lifeline ECG Vision API.

### Initialization

To start using the API, initialize the client.

*   **Key Management:** You do _not_ need an API key to generate or delete keys.
    
*   **Analysis:** You _must_ provide an API key to run the ECG analysis endpoints.
    

Python

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   from lifeline import LifelineClient  # For Key Management (No API key needed)  admin_client = LifelineClient()  # For Medical Analysis  client = LifelineClient(api_key="dasa_1234567890...")   `

### 1\. generate\_api\_key(email, admin\_secret)

Generates a new, secure API key linked to a specific user's email address. A maximum of 2 active keys are allowed per email.

*   **Parameters:**
    
    *   email _(str)_: The email address of the user registering for the key.
        
    *   admin\_secret _(str)_: The master admin password required to authorize key creation.
        
*   **Returns:** A dictionary containing the new api\_key and registration metrics.
    
*   **Raises:** requests.HTTPError if the email already has 2 keys or the admin secret is incorrect.
    

**Example Usage:**

Python

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   response = admin_client.generate_api_key(      email="doctor@hospital.com",       admin_secret="super_secret_dev_key"  )  print(f"Your new key is: {response['api_key']}")   `

### 2\. delete\_oldest\_api\_key(email, admin\_secret)

Automatically finds and deletes the oldest active API key associated with a specific email address. Useful for freeing up a slot when a user hits the 2-key registration limit.

*   **Parameters:**
    
    *   email _(str)_: The email address associated with the key you want to delete.
        
    *   admin\_secret _(str)_: The master admin password required to authorize deletion.
        
*   **Returns:** A dictionary containing the deleted key string and the number of keys remaining for that email.
    
*   **Raises:** requests.HTTPError if no active keys are found for the email.
    

**Example Usage:**

Python

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   response = admin_client.delete_oldest_api_key(      email="doctor@hospital.com",       admin_secret="super_secret_dev_key"  )  print(f"Deleted key: {response['deleted_key']}. Keys remaining: {response['keys_remaining']}")   `

### 3\. analyze(image\_path)

The core diagnostic pipeline. Uploads a raw ECG image, crops it using YOLO vision, and runs it concurrently through PULSE-7B and MedGemma to generate a synthesized, structured clinical report.

*   **Parameters:**
    
    *   image\_path _(str)_: The absolute or relative local path to the ECG image file (JPEG/PNG).
        
*   **Returns:** A dictionary containing pipeline metrics (like YOLO crop size) and the final\_report.
    
*   **Requires:** The client must be initialized with a valid api\_key.
    

**Example Usage:**

Python

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   client = LifelineClient(api_key="your_active_api_key")  try:      results = client.analyze(image_path="./patient_data/ecg_001.jpg")      print(results["final_report"])  except Exception as e:      print(f"Analysis failed: {e}")   `

### 4\. analyze\_dynamic(prompt, image\_path=None, context=None)

A highly flexible, multimodal endpoint powered by LLaVA. It allows you to ask specific questions about an ECG image, or simply query the medical LLM with text alone.

*   **Parameters:**
    
    *   prompt _(str)_: **Required.** The specific question or instruction for the AI.
        
    *   image\_path _(str, optional)_: Path to an ECG image. If provided, YOLO will crop it before sending it to the vision model.
        
    *   context _(str, optional)_: Additional patient history or clinical notes to guide the AI's final consensus.
        
*   **Returns:** A dictionary containing the modality\_used and the final\_report.
    
*   **Requires:** The client must be initialized with a valid api\_key.
    

**Example Usage:**

Python

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   client = LifelineClient(api_key="your_active_api_key")  results = client.analyze_dynamic(      prompt="Does this ECG show any signs of Atrial Fibrillation?",      image_path="./patient_data/ecg_002.png",      context="Patient is a 65-year-old male presenting with shortness of breath."  )  print(results["final_report"])   `