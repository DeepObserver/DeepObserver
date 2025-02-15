# DeepObserver (with onboard analytics)

## Running the Service

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the service:
```bash
uvicorn deepobserver.main:app --reload
```

3. Run the processor:
```bash
python deepobserver/processor.py --rtsp-url <rtsp-url> --fps <fps> --yolo-model <yolo-model> --llm-backend <llm-backend>
```

## Using Ollama with LLaVA

To use Ollama with LLaVA for image analysis:

1. Install Ollama following instructions at https://ollama.ai/

2. Pull the LLaVA model:

Note: LLaVA observations will be logged to the `logs` directory with filenames following the pattern `ollama_observations_YYYYMMDD_HHMMSS.txt`. Each session's observations and analysis results will be stored in a separate log file.