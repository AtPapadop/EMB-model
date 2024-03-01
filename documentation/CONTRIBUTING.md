# Contribution guide

## Using pull requests

1. Fork the `AtPapadop/EMB-model` repo to `<username>/EMB-model`
1. Clone the `<username>/EMB-model` locally on your workstation using `git clone`

    ```git
    git clone git@github.com:<username>/EMB-model.git
    ```
    If you dont have git installed, download it from [https://git-scm.com/download/win](https://git-scm.com/download/win)

1. By default the name of the remote in git is `origin`. This points to `<whoami>/EMB-model`. To verify use the following command inside the new directory:

    ```git
    git remote -v
    ```
1. Add a new remote (preferred name is `upstream`) using the following command:

    ```git
    git remote add upstream git@github.com:AtPapadop/EMB-model.git
    ```
1. For every pull request after cloning the `<whoami>/it-ansible` repo, sync your local git repo `main` branch by fetching the `upstream` to avoid conflict during merging:

    ```git
    git checkout main
    git fetch upstream
    git merge upstream/main 
    ```
1. Locally, checkout a new branch (i.e. `feature`)

    ```git
    git checkout -b feature
    ```

1. Do some changes, stage, commit and push them to `origin`

    ```git  
    git add path/to/changed/files
    git commit -m "Commit message"
    git push -u origin feature
    ```  
1. Open a pull request from the GitHub web interface pointing from your `origin:feature` to the `upstream:main` repo.
1. Once the pull request is tested and pulled sync your local git repo as mentioned in step 7.

1. Finally, sync your fork:

    ```git
    git push -u origin main
    ```

## SSH Access

In order to access the repo using ssh, it is required to create a private - public pair of ssh keys, add the private key to your `.ssh` folder and the public key to github.

### To create the keys

1. On windows open a powershell terminal and type the following commands:

    ```
    cd ~
    mkdir .ssh\
    cd .ssh\
    ssh-keygen
    ```

### To add the public key to gitlab

1. The public key is stored in `~\.ssh\id_rsa.pub`

     ```git
     cat ~\.ssh\id_rsa.pub
     ```
1. Copy the whole key, go to [https://github.com/settings/keys](https://github.com/settings/keys) click `New SSH key` and paste your key in.