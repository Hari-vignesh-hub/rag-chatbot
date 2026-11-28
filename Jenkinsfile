
pipeline {
    agent any

    options {
        skipDefaultCheckout(true)
    }

    stages {
        stage('Checkout') {
            steps {
                echo 'Downloading source code from GitHub...'
                checkout scm
            }
        }

        stage('Install Test Dependencies') {
            steps {
                echo 'Creating Python virtual environment...'
                sh 'python3 -m venv .venv-test'

                echo 'Installing testing dependencies...'
                sh '.venv-test/bin/pip install --upgrade pip'
                sh '.venv-test/bin/pip install -r requirements-dev.txt'
            }
        }

        stage('Run Automated Tests') {
            steps {
                echo 'Running AI RAG Chatbot tests...'
                sh '.venv-test/bin/python -m pytest tests -v'
            }
        }

        stage('Build Docker Image') {
            steps {
                echo 'Building AI RAG Chatbot Docker image...'
                sh 'docker build -t ai-rag-chatbot:${BUILD_NUMBER} -t ai-rag-chatbot:latest .'
            }
        }

        stage('Verify Docker Image') {
            steps {
                echo 'Verifying Docker image...'
                sh 'docker image inspect ai-rag-chatbot:${BUILD_NUMBER}'
            }
        }
    }

    post {
        success {
            echo 'SUCCESS: Tests passed and Docker image built successfully!'
        }

        failure {
            echo 'FAILED: Check the Jenkins console output.'
        }
    }
}
