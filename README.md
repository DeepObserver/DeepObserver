# DeepObserver (with onboard analytics)
## Running the Service

### Using Docker (Recommended)
1. Install Docker and Docker Compose on your system
2. Start the services:
```bash
# First time setup or to reset everything
docker-compose down -v
docker-compose up --build

# For subsequent runs (if you don't need to reset data)
docker-compose up -d
```

3. The services will be available at:
   - API: http://localhost:8000
   - Database: postgresql://postgres:postgres@localhost:5432/vectordb

### Local Development
1. Create and activate a Python virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:
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
example: 
```bash
python deepobserver/processor.py --rtsp-url 'rtsp://admin:georgedroyd1@192.168.5.224:554' --llm-backend ollama
```

## Using Ollama with LLaVA

To use Ollama with LLaVA for image analysis:

1. Install Ollama following instructions at https://ollama.ai/

2. Pull the LLaVA model:

Note: LLaVA observations will be logged to the `logs` directory with filenames following the pattern `ollama_observations_YYYYMMDD_HHMMSS.txt`. Each session's observations and analysis results will be stored in a separate log file.

### Useful Docker Commands
```bash
# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Reset everything (removes all data)
docker-compose down -v

# Check running containers
docker ps
```