## About the project

Flask web application that sends text to an API and by using AI determines whether it is spam or not.  

**The app returns:**

 - Spam / Not spam
 - An explanation of what influenced the decision

## Setting up the project 

**To setup the project first clone the repo:**

```bash
git clone https://github.com/c0pson/Spam-Detection-Web-App
```

**Create python virtual environment (please use at least python3.12):**

```bash
python -m venv .venv
```

**Activate virtual environment:**

 - Windows:

    ```
    .venv\Scripts\Activate.ps1
    ```

 - Unix:

    ```bash
    source .venv/bin/activate
    ```

**Install requirements:**

```bash
pip install -r requirements.txt
```

## Running application

**To start application simply type:**

```bash
python -m spam_detection
```

## How to contribute 

This prevents messy pull requests.

 1. Create a new branch for each feature or bugfix:
    ```bash
    git checkout -b feature/some-feature
    ```
 2. Follow the existing folder structure
 3. Test before pushing.
 4. Create a Pull Request to `main`

## Project structure

```bash
├───spam_detection
│   ├───api
│   ├───core
│   ├───models
│   ├───services
│   ├───static
│   │   ├───css
│   │   └───js
│   ├───templates
│   └───utils
└───tests
```

**spam_detection/**

Main application package containing all backend logic.

**api/**

Flask blueprints.
Each file inside this folder exposes HTTP endpoints such as:

 - spam detection routes
 - status/health routes
 - future API modules

**core/**

Core application setup and configuration:

 - create_app() application factory
 - config classes
 - initialization logic

**models/**

Data models and schemas used across the app, e.g.:

 - request/response objects
 - internal data structures
 - validation models (Pydantic/Marshmallow if used)

**services/**

Business logic layer. Responsible for:

 - calling the cloud spam-detection API
 - processing responses
 - internal logic independent from Flask routes

**static/**

Frontend static assets served by Flask:

 - **css/** — stylesheets for templates
 - **js/** — frontend JavaScript code
 - images or fonts (if added **img/** **fonts/**)

**templates/**

HTML templates rendered by Flask.

**utils/**

Utility functions, helpers, decorators, and reusable modules that don’t belong in services.

**tests/**

Automated test suite.
