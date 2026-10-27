
# Liu Cixin short stories

# Other resources:
# https://en.m.wikipedia.org/wiki/Liu_Cixin
# https://en.m.wikipedia.org/wiki/The_Wandering_Earth_(collection)
# https://en.m.wikipedia.org/wiki/To_Hold_Up_the_Sky
# https://en.m.wikipedia.org/wiki/A_View_from_the_Stars

books_dict = {
    "The Wandering Earth (collection)": {
        "published": 2016,
        "stories": [
            "The Wandering Earth",
            "Mountain",
            "Sun of China",
            "For the Benefit of Mankind",
            "Curse 5.0",
            "The Micro-Era",
            "Devourer",
            "Taking Care of God",
            "With Her Eyes",
            "Cannonball",
        ]
    },
    "To Hold Up the Sky": {
        "published": 2020,
        "stories": [
            "The Village Teacher",
            "Time Migration",
            "2018-04-01",
            "Fire in the Earth",
            "Contraction",
            "Mirror",
            "Ode to Joy",
            "Full Spectrum Barrage Ramming",
            "Sea of Dreams",
            "Cloud of Poems",
            "The Thinker",
        ]
    },
    "A View from the Stars": {
        "published": 2021,
        "stories": [
            "Whale Song",
            "The Messenger",
            "Butterfly",
            "End of the Microcosmos",
            "Destiny",
            "Heard It in the Morning",
            "Time Enough for Love (essay)",
            "A Journey in Search of Home (essay)",
            "Thirty Years of Making Magic Out of Ordinariness (essay)",
            "One and One Hundred Thousand Earths (essay)",
            "On Finishing Death's End (essay)",
            "The Battle Between Sci-Fi and Fantasy (essay)",
            "The 'Church' of Sci-Fi (essay)",
            "Poetic Science Fiction (essay)",
            "Civilization's Expanse in Reverse (essay)",
            "The Dark Forest Theory (essay)",
            "The World in Fifty Years (essay)",
            "On Ball Lightning (essay)",
            "We're Sci-Fi Fans (essay)",
        ]
    },
}


import sys
import os
import json
from pprint import pprint

if __name__ == "__main__":
    # Loop over books and turn dict into 1-many story to book mapping
    stories = {}
    for book in books_dict:
        for story in books_dict[book]["stories"]:
            if story not in stories:
                stories[story] = []
            stories[story].append(book)
            stories[story] = sorted(stories[story])

    print(type(stories))
    stories = dict(sorted(stories.items(), key=lambda x: x[0]))
    print(type(stories))
    
    # jstories = json.dumps(stories, indent = 4) 
    # print(jstories)
    # pprint(stories)

    print("Liu Cixin short stories - which is is in which book?\n")
    print(f"\nTotal stories: {len(stories)}\n")

    multi_books = [(story, len(stories[story]), stories[story]) for story in stories if len(stories[story]) > 1]
    print("# Stories in multiple books: (story, count, book titles)\n")
    for story, count, books in multi_books:
        print(f"{story}\t({count})\t\"{'\", \"'.join(books)}\"  ")
    if len(multi_books) == 0:
        print("None")

    single_books = [(story, len(stories[story]), stories[story]) for story in stories if len(stories[story]) == 1]
    print("\n# Stories in a single book: (story, count, book title)\n")
    for story, count, books in single_books:
        print(f"{story}\t({count})\t\"{'\", \"'.join(books)}\"  ")

    print("\n# Books with their stories:")
    for i, book in enumerate(books_dict):
        print(f"\n{i+1}. {book} ({books_dict[book]["published"]}):  ")
        for story in books_dict[book]["stories"]:
            print(f"\t- {story}  ")


# OUTPUT

# Liu Cixin short stories - which is is in which book?

# Total stories: 40

# # Stories in multiple books: (story, count, book titles)

# None

# # Stories in a single book: (story, count, book title)

# 2018-04-01 (1) "To Hold Up the Sky"  
# A Journey in Search of Home (essay) (1) "A View from the Stars"  
# Butterfly (1) "A View from the Stars"  
# Cannonball (1) "The Wandering Earth (collection)"  
# Civilization's Expanse in Reverse (essay) (1) "A View from the Stars"  
# Cloud of Poems (1) "To Hold Up the Sky"  
# Contraction (1) "To Hold Up the Sky"  
# Curse 5.0 (1) "The Wandering Earth (collection)"  
# Destiny (1) "A View from the Stars"  
# Devourer (1) "The Wandering Earth (collection)"  
# End of the Microcosmos (1) "A View from the Stars"  
# Fire in the Earth (1) "To Hold Up the Sky"  
# For the Benefit of Mankind (1) "The Wandering Earth (collection)"  
# Full Spectrum Barrage Ramming (1) "To Hold Up the Sky"  
# Heard It in the Morning (1) "A View from the Stars"  
# Mirror (1) "To Hold Up the Sky"  
# Mountain (1) "The Wandering Earth (collection)"  
# Ode to Joy (1) "To Hold Up the Sky"  
# On Ball Lightning (essay) (1) "A View from the Stars"  
# On Finishing Death's End (essay) (1) "A View from the Stars"  
# One and One Hundred Thousand Earths (essay) (1) "A View from the Stars"  
# Poetic Science Fiction (essay) (1) "A View from the Stars"  
# Sea of Dreams (1) "To Hold Up the Sky"  
# Sun of China (1) "The Wandering Earth (collection)"  
# Taking Care of God (1) "The Wandering Earth (collection)"  
# The 'Church' of Sci-Fi (essay) (1) "A View from the Stars"  
# The Battle Between Sci-Fi and Fantasy (essay) (1) "A View from the Stars"  
# The Dark Forest Theory (essay) (1) "A View from the Stars"  
# The Messenger (1) "A View from the Stars"  
# The Micro-Era (1) "The Wandering Earth (collection)"  
# The Thinker (1) "To Hold Up the Sky"  
# The Village Teacher (1) "To Hold Up the Sky"  
# The Wandering Earth (1) "The Wandering Earth (collection)"  
# The World in Fifty Years (essay) (1) "A View from the Stars"  
# Thirty Years of Making Magic Out of Ordinariness (essay) (1) "A View from the Stars"  
# Time Enough for Love (essay) (1) "A View from the Stars"  
# Time Migration (1) "To Hold Up the Sky"  
# We're Sci-Fi Fans (essay) (1) "A View from the Stars"  
# Whale Song (1) "A View from the Stars"  
# With Her Eyes (1) "The Wandering Earth (collection)"

# # Books with their stories:

# 1. The Wandering Earth (collection) (2016):  
#     - The Wandering Earth  
#     - Mountain  
#     - Sun of China  
#     - For the Benefit of Mankind  
#     - Curse 5.0  
#     - The Micro-Era  
#     - Devourer  
#     - Taking Care of God  
#     - With Her Eyes  
#     - Cannonball

# 2. To Hold Up the Sky (2020):  
#     - The Village Teacher  
#     - Time Migration  
#     - 2018-04-01  
#     - Fire in the Earth  
#     - Contraction  
#     - Mirror  
#     - Ode to Joy  
#     - Full Spectrum Barrage Ramming  
#     - Sea of Dreams  
#     - Cloud of Poems  
#     - The Thinker

# 3. A View from the Stars (2021):  
#     - Whale Song  
#     - The Messenger  
#     - Butterfly  
#     - End of the Microcosmos  
#     - Destiny  
#     - Heard It in the Morning  
#     - Time Enough for Love (essay)  
#     - A Journey in Search of Home (essay)  
#     - Thirty Years of Making Magic Out of Ordinariness (essay)  
#     - One and One Hundred Thousand Earths (essay)  
#     - On Finishing Death's End (essay)  
#     - The Battle Between Sci-Fi and Fantasy (essay)  
#     - The 'Church' of Sci-Fi (essay)  
#     - Poetic Science Fiction (essay)  
#     - Civilization's Expanse in Reverse (essay)  
#     - The Dark Forest Theory (essay)  
#     - The World in Fifty Years (essay)  
#     - On Ball Lightning (essay)  
#     - We're Sci-Fi Fans (essay)
