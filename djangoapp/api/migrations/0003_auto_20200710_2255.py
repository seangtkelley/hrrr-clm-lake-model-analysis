# Generated by Django 3.0.8 on 2020-07-10 22:55

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('api', '0002_auto_20200710_2247'),
    ]

    operations = [
        migrations.AlterField(
            model_name='ob',
            name='ob_depth',
            field=models.DecimalField(decimal_places=1, default=None, max_digits=4, null=True),
        ),
        migrations.AlterField(
            model_name='ob',
            name='sensor_name',
            field=models.CharField(default=None, max_length=256, null=True),
        ),
        migrations.AlterField(
            model_name='station',
            name='contact_email',
            field=models.CharField(default=None, max_length=256, null=True),
        ),
        migrations.AlterField(
            model_name='station',
            name='contact_name',
            field=models.CharField(default=None, max_length=256, null=True),
        ),
        migrations.AlterField(
            model_name='station',
            name='elevation',
            field=models.DecimalField(decimal_places=1, default=None, max_digits=4, null=True),
        ),
        migrations.AlterField(
            model_name='station',
            name='sensor_type',
            field=models.CharField(default=None, max_length=256, null=True),
        ),
        migrations.AlterField(
            model_name='station',
            name='water_depth',
            field=models.DecimalField(decimal_places=1, default=None, max_digits=4, null=True),
        ),
    ]
