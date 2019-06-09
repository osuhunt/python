

# Creating an XML file

import locale
import os.path
import xml.etree.cElementTree as etree
from xml.etree.cElementTree import ElementTree
from xml.etree.cElementTree import Element
from xml.etree.cElementTree import ParseError

from xml.dom import minidom

locale.setlocale(locale.LC_ALL, '')


def salary_sort(sal_record):
    """
        Used for the key= parameter to sort the player-salary data by salary
        :param sal_record: string value representing a players salary
        :return: integer value for the salary
    """
    salary = 0
    try:
        salary = int(sal_record[4])
    except ValueError:
        pass

    return salary


def get_search_year():
    """ obtain the user's input, disallow non-int values or an empty selection"""
    input_year = 1985
    while True:
        input_year = input('Search salaries for what year?--> ')
        try:
            input_year = int(input_year)
            break
        except ValueError:
            print('Invalid year, try again.')

    return input_year


def get_record_count():
    """  Returns number of records to search for """
    num_records = 10
    try:
        num_records = int(input('Number of records to retrieve (def.=10): '))
    except ValueError:
        print('Retrieving 10 records.')

    return num_records


def retrieve_data(salaries_filename, master_filename, input_year=1985, num_records=10):
    """ Works with the provided files to create the top_sals data structure """
    salaries = []
    players = {}
    top_sals = []

    try:
        # open and work with both files
        with open(salaries_filename) as file_sal, open(master_filename) as file_mast:
            for line in file_sal:                                               # get each salary record
                sal_record = line.strip().split(',')
                try:
                    record_year = int(sal_record[0])
                    if record_year == input_year:                               # only add it if the year is same as year requested
                        salaries.append(sal_record)                             # load it into a list
                except ValueError:
                    pass

            for line in file_mast:                                              # get each player record
                mast_record = line.strip().split(',')
                players[mast_record[0]] = mast_record                           # load it into a list

            salaries.sort(key=salary_sort, reverse=True)                        # sort the salary records in descending order according to salary

            for top_sal in salaries:                                            # extract necessary data from salaries file
                year = 0
                try:
                    year = int(top_sal[0])                                  # get the year for each salary record
                except ValueError:
                    pass

                try:
                    salary = float(top_sal[4])
                except ValueError:
                    salary = 0

                playerid = top_sal[3]                                       # get the player's id, salary, year
                player_data = players.get(playerid)                         # get the player's name data from the other file data structure
                if player_data:                                             # checks if the player has data in the players dictionary, if not, we ignore them
                    first_name = player_data[13]
                    last_name = player_data[14]
                    top_sals.append([first_name, last_name, salary, year])  # create a list of the player's relevant data
                    if len(top_sals) == num_records:                        # stop after 10 records
                        break
    except IOError as e:
        print('Error: {0}'.format(e))

    return top_sals


def print_report(results_filename, top_sals):
    try:
        with open(results_filename, 'w', encoding='utf8') as f_out:         # write the results to a file
            f_out.write('Results\n')
            f_out.write('{0:<40} {1:<20} {2:<8}\n'.format('Name', 'Salary', 'Year'))
            for player_data in top_sals:
                name = ' '.join(player_data[0:2])
                salary = locale.currency(int(player_data[2]), grouping=True)
                year = player_data[3]
                f_out.write('{0:<40} {1:<20} {2:<8}\n'.format(name, salary, year))
    except IOError as e:
        print(e)


def create_xml(top_sals, xml_filename):
    lroot = etree.Element('root')
    #tree = ElementTree(root)
    
    for record in top_sals:
        player = etree.Element('player')
        name = etree.Element('name')
        first_name = etree.Element('first_name')
        last_name = etree.Element('last_name')
        salary = etree.Element('salary')
        year = etree.Element('year')
        
        first_name.text = str(record[0])
        last_name.text = str(record[1])
        salary.text = str(record[2])
        year.text = str(record[3])
        
        name.append(first_name)
        name.append(last_name)
        player.append(name)
        player.append(salary)
        player.append(year)
        lroot.append(player)
        
        xml_string = etree.tostring(lroot)
        good_xml = minidom.parseString(xml_string).toprettyxml(encoding = 'utf-8')
        with open(".\data_output/xml_rsults.xml","w+") as file:
            file.write(good_xml.decode())
        file.close()
        
# Testing the functions
 
# INPUT
raw_data_dir = 'data_raw'

master_filename = 'xml_master.csv'
salaries_filename = 'xml_salaries.csv'
salfile_fullpath = os.path.join(raw_data_dir, salaries_filename)
mastfile_fullpath = os.path.join(raw_data_dir, master_filename)

# OUTPUT
output_data_dir = "data_output"
xml_filename = os.path.join(output_data_dir,'xml_results.xml')
results_filename = os.path.join(output_data_dir,'xml_results.txt')
 
 top_sals = retrieve_data(salfile_fullpath, mastfile_fullpath)
 top_sals
 """
[['Mike', 'Schmidt', 2130300.0, 1985],
 ['Gary', 'Carter', 2028571.0, 1985],
 ['George', 'Foster', 1942857.0, 1985],
 ['Dave', 'Winfield', 1795704.0, 1985],
 ['Rich', 'Gossage', 1713333.0, 1985],
 ['Dale', 'Murphy', 1625000.0, 1985],
 ['Jack', 'Clark', 1555000.0, 1985],
 ['Bob', 'Horner', 1500000.0, 1985],
 ['Eddie', 'Murray', 1472819.0, 1985],
 ['Rickey', 'Henderson', 1470000.0, 1985]]
 """
 def test_results(filename):
    try:
        with open(filename) as f_test:
            for line in f_test:
                print(line.rstrip())
    except IOError as e:
        print(e)
 
test_results(xml_filename)


<?xml version="1.0" encoding="utf-8"?>
<root>
	<player>
		<name>
			<first_name>Mike</first_name>
			<last_name>Schmidt</last_name>
		</name>
		<salary>2130300.0</salary>
		<year>1985</year>
	</player>
	<player>
		<name>
			<first_name>Gary</first_name>
			<last_name>Carter</last_name>
		</name>
		<salary>2028571.0</salary>
		<year>1985</year>
	</player>
	<player>
		<name>
			<first_name>George</first_name>
			<last_name>Foster</last_name>
		</name>
		<salary>1942857.0</salary>
		<year>1985</year>
	</player>
	<player>
		<name>
			<first_name>Dave</first_name>
			<last_name>Winfield</last_name>
		</name>
		<salary>1795704.0</salary>
		<year>1985</year>
	</player>
	<player>
		<name>
			<first_name>Rich</first_name>
			<last_name>Gossage</last_name>
		</name>
		<salary>1713333.0</salary>
		<year>1985</year>
	</player>
	<player>
		<name>
			<first_name>Dale</first_name>
			<last_name>Murphy</last_name>
		</name>
		<salary>1625000.0</salary>
		<year>1985</year>
	</player>
	<player>
		<name>
			<first_name>Jack</first_name>
			<last_name>Clark</last_name>
		</name>
		<salary>1555000.0</salary>
		<year>1985</year>
	</player>
	<player>
		<name>
			<first_name>Bob</first_name>
			<last_name>Horner</last_name>
		</name>
		<salary>1500000.0</salary>
		<year>1985</year>
	</player>
	<player>
		<name>
			<first_name>Eddie</first_name>
			<last_name>Murray</last_name>
		</name>
		<salary>1472819.0</salary>
		<year>1985</year>
	</player>
	<player>
		<name>
			<first_name>Rickey</first_name>
			<last_name>Henderson</last_name>
		</name>
		<salary>1470000.0</salary>
		<year>1985</year>
	</player>
</root>

# It works!
