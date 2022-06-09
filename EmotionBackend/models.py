from django.db import models


#Person Model
class Person(models.Model):
    name= models.CharField(max_length=200,null=False)
    username=models.CharField(max_length=100,unique=True,null=False)
    email=models.CharField(max_length=50,unique=True,null=False)
    password=models.CharField(max_length=50,null=False)

    def __str__(self):
        return str(self.id) + '-' + self.username

def upload_path(instance,filename):
    return "/".join(['image',str(instance),filename])

# Image Class
class Image(models.Model):
    image=models.FileField(null=False,upload_to=upload_path)

    def __str__(self):
        return str(self.id)