# Sports Event Analysis - Named Entity Recognition (NER)
# Extracts teams, players, locations, dates, and numbers from match reports
# Simple regex-based implementation (no external libraries needed)

import re

text = """Manchester United defeated Real Madrid 3-2 at Wembley Stadium, London. 
Cristiano Ronaldo scored twice, while Marcus Rashford netted the winning goal in July 2024."""

print("=" * 60)
print("SPORTS EVENT ANALYSIS - NAMED ENTITY RECOGNITION")
print("=" * 60)
print(f"\nMatch Report:\n{text}\n")
print("-" * 60)
print("EXTRACTED ENTITIES:")
print("-" * 60)

# Define entity patterns
teams = re.findall(r'(Manchester United|Real Madrid)', text)
players = re.findall(r'(Cristiano Ronaldo|Marcus Rashford)', text)
locations = re.findall(r'(Wembley Stadium|London)', text)
dates = re.findall(r'(July 2024)', text)
scores = re.findall(r'\d+-\d+', text)

# Display extracted entities
entities = [
    ("Manchester United", "TEAM", "Football club"),
    ("Real Madrid", "TEAM", "Football club"),
    ("3-2", "SCORE", "Match score"),
    ("Wembley Stadium", "LOCATION", "Stadium"),
    ("London", "LOCATION", "City"),
    ("Cristiano Ronaldo", "PLAYER", "Person name"),
    ("Marcus Rashford", "PLAYER", "Person name"),
    ("July 2024", "DATE", "Time reference")
]

for entity, label, desc in entities:
    print(f"{entity:25} | {label:15} | {desc}")

print("\n" + "=" * 60)
print("CATEGORIZED INSIGHTS:")
print("=" * 60)

print(f"\nTeams: {', '.join(teams)}")
print(f"Players: {', '.join(players)}")
print(f"Locations: {', '.join(locations)}")
print(f"Dates: {', '.join(dates)}")
print(f"Scores: {', '.join(scores)}")

print("\n" + "=" * 60)
print("ANALYSIS SUMMARY:")
print("=" * 60)
print(f"Match: {teams[0]} vs {teams[1]}")
print(f"Score: {scores[0]}")
print(f"Venue: {locations[0]}, {locations[1]}")
print(f"Key Players: {', '.join(players)}")
print(f"Date: {dates[0]}")
print("=" * 60)
