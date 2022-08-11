class Server:
    def __init__(self) -> None:
        self.totalSentPackages = 0
        self.packagesList = []
    
    def updateSentPackages(self, package):
        self.packagesList.append(package)
        self.totalSentPackages += 1
    
    def getPackagesList(self):
        return self.packagesList
    
    def getTotalPackages(self):
        return self.totalSentPackages