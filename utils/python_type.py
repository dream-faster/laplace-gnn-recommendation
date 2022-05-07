import os


def isnotebook():
    if "COLAB_GPU" in os.environ:
        print("I'm running on Colab")
        return True
    else:
        try:
            shell = get_ipython().__class__.__name__
            print(shell)
            if shell == "ZMQInteractiveShell":
                return True  # Jupyter notebook or qtconsole
            elif shell == "google.colab._shell":
                return True  # Google Colab Shell
            elif shell == "TerminalInteractiveShell":
                return False  # Terminal running IPython
            else:
                return False  # Other type (?)
        except NameError:
            return False  # Probably standard Python interpreter
