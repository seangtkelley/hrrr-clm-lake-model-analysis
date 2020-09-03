from django.core.management.base import BaseCommand, CommandError

from lib import utils

class Command(BaseCommand):
    help = 'Loads data from 2020-HRRRxCLM-SPoRT-Insitu-Points-Matchup.csv into database'

    def handle(self, *args, **options):
        utils.load_metadata_into_db()

        self.stdout.write(self.style.SUCCESS("Successfully loaded metadata"))
