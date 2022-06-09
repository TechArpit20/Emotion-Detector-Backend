from rest_framework import serializers

from EmotionBackend.models import Person, Image


class PersonSerializer(serializers.ModelSerializer):
    class Meta:
        model=Person
        exclude=('password',)

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model=Person
        fields="__all__"

class ImageSerializer(serializers.ModelSerializer):
    class Meta:
        model=Image
        fields="__all__"
