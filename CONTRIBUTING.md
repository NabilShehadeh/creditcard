# Contributing to Credit Card Fraud Detection

Thank you for your interest in contributing to the Credit Card Fraud Detection project! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Guidelines](#contributing-guidelines)
- [Pull Request Process](#pull-request-process)
- [Code Style](#code-style)
- [Testing](#testing)
- [Documentation](#documentation)
- [Issue Reporting](#issue-reporting)

## Code of Conduct

This project adheres to a code of conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/credit-card-fraud-detection.git
   cd credit-card-fraud-detection
   ```
3. **Add the upstream repository**:
   ```bash
   git remote add upstream https://github.com/original-owner/credit-card-fraud-detection.git
   ```

## Development Setup

### Prerequisites

- Python 3.8 or higher
- pip or conda
- Git

### Environment Setup

1. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # Development dependencies
   ```

3. **Install pre-commit hooks** (optional but recommended):
   ```bash
   pre-commit install
   ```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_fraud_detection.py

# Run with verbose output
pytest -v
```

### Running the API

```bash
# Development mode
python api/main.py

# Using Docker
docker-compose up --build
```

## Contributing Guidelines

### Types of Contributions

We welcome several types of contributions:

- **Bug fixes**: Fix issues in existing code
- **Feature additions**: Add new functionality
- **Documentation**: Improve or add documentation
- **Tests**: Add or improve test coverage
- **Performance improvements**: Optimize existing code
- **Code refactoring**: Improve code structure and readability

### Before You Start

1. **Check existing issues** to see if your contribution is already being worked on
2. **Create an issue** for significant changes to discuss the approach
3. **For small fixes**, you can directly create a pull request

### Development Workflow

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b bugfix/issue-description
   ```

2. **Make your changes** following the code style guidelines

3. **Add tests** for new functionality

4. **Update documentation** if needed

5. **Run tests** to ensure everything works:
   ```bash
   pytest
   black --check src/ tests/ api/
   flake8 src/ tests/ api/
   ```

6. **Commit your changes**:
   ```bash
   git add .
   git commit -m "Add: brief description of changes"
   ```

7. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

8. **Create a Pull Request** on GitHub

## Pull Request Process

### PR Requirements

- [ ] Code follows the project's style guidelines
- [ ] Self-review of code has been performed
- [ ] Tests have been added/updated and pass
- [ ] Documentation has been updated if necessary
- [ ] PR description clearly explains the changes
- [ ] All CI checks pass

### PR Description Template

```markdown
## Description
Brief description of the changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Testing
- [ ] Tests pass locally
- [ ] New tests added for new functionality
- [ ] Manual testing performed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or documented if intentional)
```

## Code Style

### Python Code Style

We follow PEP 8 with some modifications:

- **Line length**: 100 characters maximum
- **Import order**: Standard library, third-party, local imports
- **Docstrings**: Google style docstrings for functions and classes

### Formatting Tools

We use automated formatting tools:

- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting

Run formatting before committing:
```bash
black src/ tests/ api/
isort src/ tests/ api/
flake8 src/ tests/ api/
```

### Code Examples

**Good**:
```python
def calculate_fraud_probability(transaction_data: Dict[str, float]) -> float:
    """
    Calculate fraud probability for a transaction.
    
    Args:
        transaction_data: Dictionary containing transaction features
        
    Returns:
        Fraud probability between 0 and 1
    """
    # Implementation here
    pass
```

**Bad**:
```python
def calc_fraud_prob(data):  # No type hints, unclear name
    # No docstring
    pass
```

## Testing

### Test Structure

- **Unit tests**: Test individual functions and classes
- **Integration tests**: Test component interactions
- **API tests**: Test API endpoints
- **Model tests**: Test ML model functionality

### Writing Tests

```python
def test_data_processor_load_data():
    """Test data loading functionality."""
    processor = DataProcessor()
    df = processor.load_data()
    
    assert len(df) > 0
    assert 'Class' in df.columns
    assert df['Amount'].min() >= 0
```

### Test Coverage

We aim for >80% test coverage. Run coverage analysis:

```bash
pytest --cov=src --cov-report=html
open htmlcov/index.html  # View coverage report
```

## Documentation

### Documentation Types

- **Code documentation**: Docstrings and comments
- **API documentation**: FastAPI auto-generated docs
- **README**: Project overview and setup
- **Model card**: ML model documentation
- **Notebooks**: Jupyter notebooks for analysis

### Writing Documentation

- Use clear, concise language
- Include code examples
- Update documentation when code changes
- Follow the existing documentation style

## Issue Reporting

### Bug Reports

When reporting bugs, include:

- **Environment**: Python version, OS, dependencies
- **Steps to reproduce**: Clear, numbered steps
- **Expected behavior**: What should happen
- **Actual behavior**: What actually happens
- **Error messages**: Full error traceback if applicable
- **Screenshots**: If relevant

### Feature Requests

For feature requests, include:

- **Use case**: Why is this feature needed?
- **Proposed solution**: How should it work?
- **Alternatives**: Other approaches considered
- **Additional context**: Any other relevant information

## Getting Help

- **Documentation**: Check the README and docs/
- **Issues**: Search existing issues
- **Discussions**: Use GitHub Discussions for questions
- **Contact**: Reach out to maintainers

## Recognition

Contributors will be recognized in:

- **CONTRIBUTORS.md**: List of all contributors
- **Release notes**: Major contributors mentioned
- **GitHub**: Contributor statistics

## License

By contributing to this project, you agree that your contributions will be licensed under the same license as the project (MIT License).

---

Thank you for contributing to the Credit Card Fraud Detection project! 🚀
