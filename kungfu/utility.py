def find_csv_filenames(path_to_dir, extension = ".csv"):

    '''
    returns list of filenames with chosen extension in chosen path
    '''

    filenames = os.listdir(path_to_dir)
    return [filename for filename in filenames if filename.endswith(extension)]


def get_unique_from_list(list):

    '''
    returns a list of unique values in a list
    '''

    array = np.array(list)
    return np.unique(array).tolist()


def add_months_to_date(date, delta_months=1):

    '''
    returns the date of the first day of the month delta_months months ahead
    '''

    for i in range(0,delta_months):
        date = (date.replace(day=1) + dt.timedelta(days=32)).replace(day=1)
    return date
