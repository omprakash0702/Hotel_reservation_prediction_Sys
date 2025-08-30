pipeline{
    agent any
    stages{
        stage('Cloning repository'){
            steps{
                script{
                    echo "Clonning repository"
                    checkout scmGit(branches: [[name: '*/main']], extensions: [], userRemoteConfigs: [[credentialsId: 'github-token', url: 'https://github.com/omprakash0702/Hotel_reservation_prediction_Sys.git']])
                }
            }
        }
    }
}