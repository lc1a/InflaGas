# Generated by Django 4.0.4 on 2022-06-24 21:49

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('homepage', '0004_auto_20220624_1153'),
    ]

    operations = [
        migrations.CreateModel(
            name='TesteParams',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('Algoritmo', models.CharField(max_length=30)),
                ('Parametro', models.CharField(max_length=10)),
                ('Valores', models.FloatField()),
                ('R2Treino', models.FloatField()),
                ('R2Teste', models.FloatField()),
            ],
        ),
    ]