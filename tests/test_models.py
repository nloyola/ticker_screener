import time

from app.models import Sector, Subsector


def test_created_at_is_taken_when_each_object_is_made():
    first_sector = Sector(id=1, sector='Energy')
    first_subsector = Subsector(id=1, sector_id=1, subsector='Oil', tickers='XOM')
    time.sleep(0.01)
    second_sector = Sector(id=2, sector='Tech')
    second_subsector = Subsector(id=2, sector_id=2, subsector='Chips', tickers='NVDA')

    assert second_sector.created_at > first_sector.created_at
    assert second_subsector.created_at > first_subsector.created_at
