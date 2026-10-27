
# Greg Egan short stories

# Other resources:
# https://www.gregegan.net/
# https://kasmana.people.cofc.edu/MATHFICT/readinglists.php  (Maths fiction recommendations)

books_dict = {
    "The Best of Greg Egan": {
        "goodreads": "https://goodreads.com/en/book/show/44563837",
        "amazon": "https://www.amazon.com.au/Best-Greg-Egan-S-F-MASTERWORKS-ebook/dp/B08GX88T4J",
        "metadata": "stories from goodreads link",
        "pages": 622,
        "kindle_price_au": 12.99,
        "paperback_price_au": 19.25,
        "published": 2021,
        "stories": [
            "Learning to Be Me",
            "Axiomatic",
            "Appropriate Love",
            "Into Darkness",
            "Unstable Orbits in the Space of Lies",
            "Closer",  # https://web.archive.org/web/20110723113652/http://eidolon.net/?story=Closer
            "Chaff",
            "Luminous",
            "Silver Fire",
            "Reasons to be Cheerful",
            "Oceanic",
            "Oracle",
            "Singleton",
            "Dark Integers",
            "Crystal Nights",
            "Zero For Conduct",
            "Bit Players",
            "Uncanny Valley",
            "3-adica",
            "Instantiation",
            "Afterword",
        ]
    },
    "Sleep and the Soul": {
        "amazon": "https://www.amazon.com.au/Sleep-Soul-Greg-Egan/dp/192224046X",
        "pages": 382,
        "kindle_price_au": 5.74,
        "published": 2023,
        "stories": [
            "You and Whose Army?",
            "This Is Not the Way Home",
            "Zeitgeber",
            "Crisis Actors",
            "Sleep and the Soul",
            "After Zero",
            "Dream Factory",
            "Light Up the Clouds",
            "Night Running",
            "Solidity",
        ]
    },
    "Instantiation": {
        "amazon": "https://www.amazon.com.au/Instantiation-Greg-Egan/dp/1922240338",
        "pages": 436,
        "kindle_price_au": 5.70,
        "published": 2020,
        "stories": [
            "The Discrete Charm of the Turing Machine",
            "Zero For Conduct",
            "Uncanny Valley",
            "Seventh Sight",
            "The Nearest",
            "Shadow Flock",
            "Bit Players",
            "Break My Fall",
            "3-adica",
            "The Slipway",
            "Instantiation",
        ]
    },
    "Axiomatic: Short Stories of Science Fiction": {
        "amazon": "https://www.amazon.com.au/Axiomatic-Short-Stories-Science-Fiction/dp/1597805408",
        "goodreads": "https://www.goodreads.com/book/show/156783.Axiomatic",
        "pages": 384,
        "published": 1990,
        "stories": [
            "The Infinite Assassin",
            "The Hundred-Light-Year Diary",
            "Eugene",
            "The Caress",
            "Blood Sisters",
            "Axiomatic",
            "The Safe-Deposit Box",
            "Seeing",
            "The Moat",
            "The Cutie",
            "Into Darkness",
            "Appropriate Love",
            "The Moral Virologist",
            "Unstable Orbits in the Space Of Lies",
        ]
    },
    "Phoresis and Other Journeys": {
        "amazon": "https://www.amazon.com.au/Phoresis-Other-Journeys-Greg-Egan-ebook/dp/B0C4JJ8K7Q",
        "published": 2023,
        "pages": 301,
        "kindle_price_au": 5.91,
        "stories": [
            "The Four Thousand, the Eight Hundred", 
            "Dispersion", 
            "Phoresis"],
    },
    "Luminous": {
        "amazon": "https://www.amazon.com.au/Luminous-Greg-Egan-ebook/dp/B004JHY82Q/",
        "wiki": "https://en.wikipedia.org/wiki/Luminous_(book)",
        "published": 2010,
        "kindle_price_au": 13.99,
        "stories": [
            "Chaff",
            "Mitochondrial Eve",
            "Luminous",
            "Mister Volition",
            "Cocoon",
            "Transition Dreams",
            "Silver Fire",
            "Reasons to Be Cheerful",
            "Our Lady of Chernobyl",
            "The Planck Dive",
        ]
    },
    "Dark Integers and Other Stories": {
        "amazon": "https://www.amazon.com.au/dp/1596061553",
        "paperback_price_au": 95.75,
        "kindle_price_au": -1,
        "published": 2009,
        "metadata": "out of print? very expensive!",
        "stories": [
            "Luminous", 
            "Riding the Crocodile", 
            "Dark Integers", 
            "Glory", 
            "Oceanic",
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

    print("Greg Egan short stories - which is is in which book?\n")
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

    # import pdb; pdb.set_trace()


# OUTPUT:

# Greg Egan short stories - which is is in which book?

# Total stories: 60

# # Stories in multiple books: (story, count, book titles)

# 3-adica (2) "Instantiation", "The Best of Greg Egan"  
# Appropriate Love (2) "Axiomatic: Short Stories of Science Fiction", "The Best of Greg Egan"  
# Axiomatic (2) "Axiomatic: Short Stories of Science Fiction", "The Best of Greg Egan"  
# Bit Players (2) "Instantiation", "The Best of Greg Egan"  
# Chaff (2) "Luminous", "The Best of Greg Egan"  
# Dark Integers (2) "Dark Integers and Other Stories", "The Best of Greg Egan"  
# Instantiation (2) "Instantiation", "The Best of Greg Egan"  
# Into Darkness (2) "Axiomatic: Short Stories of Science Fiction", "The Best of Greg Egan"  
# Luminous (3) "Dark Integers and Other Stories", "Luminous", "The Best of Greg Egan"  
# Oceanic (2) "Dark Integers and Other Stories", "The Best of Greg Egan"  
# Silver Fire (2) "Luminous", "The Best of Greg Egan"  
# Uncanny Valley (2) "Instantiation", "The Best of Greg Egan"  
# Zero For Conduct (2) "Instantiation", "The Best of Greg Egan"

# # Stories in a single book: (story, count, book title)

# After Zero (1) "Sleep and the Soul"  
# Afterword (1) "The Best of Greg Egan"  
# Blood Sisters (1) "Axiomatic: Short Stories of Science Fiction"  
# Break My Fall (1) "Instantiation"  
# Closer (1) "The Best of Greg Egan"  
# Cocoon (1) "Luminous"  
# Crisis Actors (1) "Sleep and the Soul"  
# Crystal Nights (1) "The Best of Greg Egan"  
# Dispersion (1) "Phoresis and Other Journeys"  
# Dream Factory (1) "Sleep and the Soul"  
# Eugene (1) "Axiomatic: Short Stories of Science Fiction"  
# Glory (1) "Dark Integers and Other Stories"  
# Learning to Be Me (1) "The Best of Greg Egan"  
# Light Up the Clouds (1) "Sleep and the Soul"  
# Mister Volition (1) "Luminous"  
# Mitochondrial Eve (1) "Luminous"  
# Night Running (1) "Sleep and the Soul"  
# Oracle (1) "The Best of Greg Egan"  
# Our Lady of Chernobyl (1) "Luminous"  
# Phoresis (1) "Phoresis and Other Journeys"  
# Reasons to Be Cheerful (1) "Luminous"  
# Reasons to be Cheerful (1) "The Best of Greg Egan"  
# Riding the Crocodile (1) "Dark Integers and Other Stories"  
# Seeing (1) "Axiomatic: Short Stories of Science Fiction"  
# Seventh Sight (1) "Instantiation"  
# Shadow Flock (1) "Instantiation"  
# Singleton (1) "The Best of Greg Egan"  
# Sleep and the Soul (1) "Sleep and the Soul"  
# Solidity (1) "Sleep and the Soul"  
# The Caress (1) "Axiomatic: Short Stories of Science Fiction"  
# The Cutie (1) "Axiomatic: Short Stories of Science Fiction"  
# The Discrete Charm of the Turing Machine (1) "Instantiation"  
# The Four Thousand, the Eight Hundred (1) "Phoresis and Other Journeys"  
# The Hundred-Light-Year Diary (1) "Axiomatic: Short Stories of Science Fiction"  
# The Infinite Assassin (1) "Axiomatic: Short Stories of Science Fiction"  
# The Moat (1) "Axiomatic: Short Stories of Science Fiction"  
# The Moral Virologist (1) "Axiomatic: Short Stories of Science Fiction"  
# The Nearest (1) "Instantiation"  
# The Planck Dive (1) "Luminous"  
# The Safe-Deposit Box (1) "Axiomatic: Short Stories of Science Fiction"  
# The Slipway (1) "Instantiation"  
# This Is Not the Way Home (1) "Sleep and the Soul"  
# Transition Dreams (1) "Luminous"  
# Unstable Orbits in the Space Of Lies (1) "Axiomatic: Short Stories of Science Fiction"  
# Unstable Orbits in the Space of Lies (1) "The Best of Greg Egan"  
# You and Whose Army? (1) "Sleep and the Soul"  
# Zeitgeber (1) "Sleep and the Soul"

# # Books with their stories:

# 1. The Best of Greg Egan (2021):  
#     - Learning to Be Me  
#     - Axiomatic  
#     - Appropriate Love  
#     - Into Darkness  
#     - Unstable Orbits in the Space of Lies  
#     - Closer  
#     - Chaff  
#     - Luminous  
#     - Silver Fire  
#     - Reasons to be Cheerful  
#     - Oceanic  
#     - Oracle  
#     - Singleton  
#     - Dark Integers  
#     - Crystal Nights  
#     - Zero For Conduct  
#     - Bit Players  
#     - Uncanny Valley  
#     - 3-adica  
#     - Instantiation  
#     - Afterword

# 2. Sleep and the Soul (2023):  
#     - You and Whose Army?  
#     - This Is Not the Way Home  
#     - Zeitgeber  
#     - Crisis Actors  
#     - Sleep and the Soul  
#     - After Zero  
#     - Dream Factory  
#     - Light Up the Clouds  
#     - Night Running  
#     - Solidity

# 3. Instantiation (2020):  
#     - The Discrete Charm of the Turing Machine  
#     - Zero For Conduct  
#     - Uncanny Valley  
#     - Seventh Sight  
#     - The Nearest  
#     - Shadow Flock  
#     - Bit Players  
#     - Break My Fall  
#     - 3-adica  
#     - The Slipway  
#     - Instantiation

# 4. Axiomatic: Short Stories of Science Fiction (1990):  
#     - The Infinite Assassin  
#     - The Hundred-Light-Year Diary  
#     - Eugene  
#     - The Caress  
#     - Blood Sisters  
#     - Axiomatic  
#     - The Safe-Deposit Box  
#     - Seeing  
#     - The Moat  
#     - The Cutie  
#     - Into Darkness  
#     - Appropriate Love  
#     - The Moral Virologist  
#     - Unstable Orbits in the Space Of Lies

# 5. Phoresis and Other Journeys (2023):  
#     - The Four Thousand, the Eight Hundred  
#     - Dispersion  
#     - Phoresis

# 6. Luminous (2010):  
#     - Chaff  
#     - Mitochondrial Eve  
#     - Luminous  
#     - Mister Volition  
#     - Cocoon  
#     - Transition Dreams  
#     - Silver Fire  
#     - Reasons to Be Cheerful  
#     - Our Lady of Chernobyl  
#     - The Planck Dive

# 7. Dark Integers and Other Stories (2009):  
#     - Luminous  
#     - Riding the Crocodile  
#     - Dark Integers  
#     - Glory  
#     - Oceanic
