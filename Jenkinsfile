pipeline{
    agent any


    Environment {
        VENV_DIR = 'venv'
    }

    stages{
        stage('Cloning Github Repo Jenkins'){
            steps{
                script{
                    echo 'Cloning Github repo to Jenkins......................... '
                    checkout scmGit(branches: [[name: '*/main']], extensions: [], userRemoteConfigs: [[credentialsId: 'github-token', url: 'https://github.com/Prajwalpatelp/Hotel_Reservations_Project.git']])
                }
            }
        }
        stages{
        stage('Setting up our virtual Environment and Installing dependancies'){
            steps{
                script{
                    echo 'Setting up our virtual Environment and Installing dependancies......................... '
                    sh '''
                    python -m venv ${VENV_DIR}
                    .${VENV_DIR}/bin/activate
                    pip install --upgrade pip
                    pip install -e .


                    '''
                }
            }
        }
    }
}