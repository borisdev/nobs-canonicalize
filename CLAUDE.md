# CLAUDE.md - Guidelines for nobs-clusters

## Build & Test Commands
```bash
# Install dependencies
poetry install

# Run all tests
poetry run pytest 

# Run a specific test
poetry run pytest tests/test_main.py::test_embeddings -v

# Run with specific markers
poetry run pytest -v
```

## Code Style & Conventions
- **Imports**: stdlib → third-party → project; separate with blank lines
- **Types**: Use type hints for parameters and return values (`-> Type`)
- **Naming**: snake_case for functions/variables, PascalCase for classes
- **Models**: Use Pydantic for data validation and models
- **Error handling**: Specific exceptions, descriptive messages
- **Documentation**: Triple-quoted docstrings on classes, descriptive function names
- **Testing**: Use pytest with fixtures, name tests with `test_` prefix

## Project Architecture
- Uses BERTopic for topic modeling with OpenAI embeddings
- Supports both regular OpenAI and Azure OpenAI clients
- Implements disk caching for performance optimization