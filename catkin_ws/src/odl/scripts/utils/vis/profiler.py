import json
from time import time

class Profiler:
    """
    Creates a profiler object that tracks the execution time of a block of code.
    The standard pipeline to record the time needed for code execution is as follows.

    1. Declare an instance of the Profiler class
    2. Create a "tracking" item. Give a relevant name of the task 
    that will measure that time for ( e.g profiler.addTrackItem("inference"))
    3. Start and stop the timer using the corresponding member 
    functions. ( e.g profiler.start("inference"); profiler.stop("inference"))
    4. Write the profiling to a json file. ( e.g profiler.write_json_report(filename))
    """
    def __init__(self) -> None:
        self.items = []
        self.status = {}

    def addTrackItem(self,itemName: str) -> None:
        """Adds a new tracking element. Try to use a name that is relevant 
        to the the code that the timer will measure the duration. (e.g. post-processing)

        If an item with the name itemName already exists then it's contents are cleared.
        Args:
            itemName (str): Name of code block alias
        """
        if itemName not in self.items and itemName not in self.status:
            self.items.append(itemName)
            print(f"{itemName} doesn't exist on self.items")
        else:
            self.status[itemName].clear()
    def start(self,itemName : str) -> None:
        """Starts the timer for the item ItemName

        Args:
            itemName (str): Name of code blovk alias

        Raises:
            Exception: If an item is not added through the addTrackItem function.
        """
        start_time = time()
        if itemName in self.items:
            self.status[itemName] = {'started':start_time}
        else:
            raise Exception(f"The item {itemName} is not defined as an item to be tracked.\
                            Please add it using addTrackItem()")
    def stop(self,itemName: str) -> float:
        """Starts the timer for the item ItemName

        Args:
            itemName (str): Name of code blovk alias

        Raises:
            Exception: If an item is not added through the addTrackItem function.

        Returns:
            float: The measured time difference between 
            profiler.start and profiler.stop
        """
        stop_time = time()
        if itemName in self.items:
            self.status[itemName]['stoped'] = stop_time
            self.status[itemName]['time'] = self.status[itemName]['stoped'] \
                - self.status[itemName]['started']
            return self.status[itemName]['time']
        else:
            raise Exception(f"The item {itemName} is not defined as an item to be tracked.\
                            Please add it using addTrackItem()")
        
    def write_json_report(self,file : str):
        """Write profiler.status in a json report file.

        Args:
            file (str): Path to new json file.
        """
        with open(file,'w') as f:
            profiling = json.dumps(self.status,indent=4)
            f.write(profiling)
    
    def deleteTrackItem(self,itemName :str):
        """Deletes a track item from profiler.items and profiler.status

        Args:
            itemName (str): Name of code block alias
        """
        self.items.remove(itemName)
        self.status.pop(itemName)
