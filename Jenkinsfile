pipeline {
    agent any

    environment {
        VENV_DIR = 'venv'
    }

    stages {

        stage('Cloning Github Repo Jenkins') {
            steps {
                script {
                    echo 'Cloning Github repo to Jenkins......................... '
                    checkout scmGit(
                        branches: [[name: '*/main']],
                        extensions: [],
                        userRemoteConfigs: [[
                            credentialsId: 'github-token',
                            url: 'https://github.com/Prajwalpatelp/Hotel_Reservations_Project.git'
                        ]]
                    )
                }
            }
        }

        stage('Setting up Virtual Environment and Installing Dependencies') {
            steps {
                script {
                    echo 'Setting up virtual environment and installing dependencies......................... '
                    sh '''
                    python -m venv ${VENV_DIR}
                    . ${VENV_DIR}/bin/activate
                    pip install --upgrade pip
                    pip install -e .
                    '''
                }
            }
        }

    }
}
