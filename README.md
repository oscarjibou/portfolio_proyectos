# College Projects from Telecomunication Engineering and Business Administration
## Matlab Set Up:
    - First Matlab should be installed and usable in the Applications folder
    - Then compile the command: . ./setup.sh
    - For being able to compile the matlab file, use the command: make mtlb run="filename"  //This file should have the .m extension
    - Now your terminal will be matlab, so if you want to compile any file inside the TDS folder just type: "filename"
    - For exiting this matlab terminal just type: exit

## Virtual Python environment:
    Installation: pip install virtualenv OR pip3 install virtualenv
    Creation: python3 -m venv venv
    Activation:
        For Mac: source venv/bin/activate
        For Windows: .\venv\Scripts\activate
    Requirements: pip install -r requirements.txt
    Updating requirements: pip freeze > requirements.txt


## Changes and staged changes guide:
    Once you have made a change in the file and save it, it will appear as a change in Source Control.
    There you can review the changes in a Working Tree Window, where you can undo big changes.
    When the changes you have made work the way you like it, you should click on the + symbol, that will move it to Stagged Changes.
    This way you can separate between different versions, and avoid making undoable changes. 

## GitHub Workflow:
    Go to dev, then create a new branch with: git checkout -b "name of the new branch"
    Work in the branch until it works perfectly, then upload it to git with:
        For staged changes: git commit -m "comment explaining the changes"      Then: git push
        For all changes: git commit -am "comment explaining the changes"        Then: git push
    Once your job is finished, create a pull request to dev. Someone else will have to check it and approve it.
    After all changes are tested and working in dev, we will merg it with main.


## Accessing Encrypted Files

Certain files in this repository are encrypted for security reasons. To access these encrypted files, please follow the steps below:

### 1. Contact the Repository Owner

To gain access to the encrypted files, you need to contact the repository owner to obtain the public GPG key.

**Contact Information:**

- **Name**: Oscar
- **Email**: oscarjibou@gmail.com

Send an email requesting access to the encrypted files. The repository owner will provide you with a public GPG key file named `publickey.asc`.

### 2. Install GPG

If you don't already have GPG installed on your system, you can install it using Homebrew.

#### Installing Homebrew (if not already installed)

First, install Homebrew by running the following command in your terminal:

```sh
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

#### Installing GPG

After installing Homebrew, install GPG with the following command:

```sh
brew install gnupg
```

### 3. Import the Public GPG Key

Once you receive the `publickey.asc` file, you need to import it into your GPG keyring.

```sh
gpg --import publickey.asc
```

This command will add the repository owner's public key to your GPG keyring, allowing you to decrypt the files.

### 4. Clone the Repository

Clone this repository to your local machine if you haven't already.

```sh
git clone https://github.com/username/your-repo.git
cd your-repo
```

### 5. Unlock the Repository with `git-crypt`

With the public GPG key imported, you can now unlock the repository using `git-crypt`.

```sh
git-crypt unlock
```

This command will use the GPG key to decrypt the encrypted files in the repository.

### Example Workflow

Here is a step-by-step example workflow:

1. **Contact the Repository Owner**:
   - Send an email to `carlosedm@gmail.com` requesting access to the encrypted files.

2. **Receive and Import the Public GPG Key**:
   - Save the `publickey.asc` file received from the repository owner.
   - Import the public GPG key:
     ```sh
     gpg --import publickey.asc
     ```

3. **Install GPG (if not already installed)**:
   - Install Homebrew (if not already installed):
     ```sh
     /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
     ```
   - Install GPG:
     ```sh
     brew install gnupg
     ```

4. **Clone the Repository**:
   - Clone the repository to your local machine:
     ```sh
     git clone https://github.com/username/your-repo.git
     cd your-repo
     ```

5. **Unlock the Repository**:
   - Unlock the repository to decrypt the files:
     ```sh
     git-crypt unlock
     ```

### Troubleshooting

If you encounter any issues during the process, please ensure the following:

- You have GPG installed on your system.
- You have the correct public GPG key from the repository owner.
- You have `git-crypt` installed on your system.
