# Generated by Django 4.0.5 on 2022-06-07 18:23

import EmotionBackend.models
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('EmotionBackend', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='Image',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('image', models.FileField(upload_to=EmotionBackend.models.upload_path)),
            ],
        ),
    ]
