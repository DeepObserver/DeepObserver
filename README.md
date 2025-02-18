# DeepObserver
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

### Local Development (Alternative)
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the service:
```bash
uvicorn deepobserver.main:app --reload
```

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