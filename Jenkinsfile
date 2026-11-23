
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

        stage('Build Docker Image') {
            steps {
                echo 'Building AI RAG Chatbot Docker image...'
                sh 'docker build -t ai-rag-chatbot:${BUILD_NUMBER} -t ai-rag-chatbot:latest .'
            }
        }

        stage('Verify Docker Image') {
            steps {
                echo 'Verifying the Docker image...'
                sh 'docker image inspect ai-rag-chatbot:${BUILD_NUMBER}'
            }
        }
    }

    post {
        success {
            echo 'SUCCESS: AI RAG Chatbot Docker image built successfully!'
        }

        failure {
            echo 'FAILED: Please check the Jenkins console output.'
        }
    }
}
