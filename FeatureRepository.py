import os
import sys
import imp
import re
import shutil
import pandas as pd
import numpy as np
import numbers
import datetime
import hashlib
import pickle
from functools import reduce


class DefaultExtractor:
    ''' Example of Extractor, input: list of numpy ndarray, out put ndarray.
    '''
    @classmethod
    def extract(self, cleandata):
        feature = np.concatenate(cleandata, axis=1)
        # put your code here
        return feature


class Repository:
    @classmethod
    def help(cls, self, detail=1):
        print(''' Methods:
create_repo, set_path, load_repo, delete_repo
add_raw_data, get_raw_data, delete_raw_data
add_clean_data, get_clean_data
add_extractor, log_extractor, get_extractor, get_extractor_str
extract_feature, add_feature, get_feature, extract_feature
    Please use methodname_() to checkout the usage.
        ''')

        if(detail >= 2):
            print('''
Columns in rawdata_info.csv
    Name: name for the raw dataframe (only _ and - are allowed, other special
        char is not valid)
    Rows: row count of the raw data, ie sample size
    Cols: column count of the raw data, ie raw feature count
    DtTm: date time marker
    Notes: other descriptions of the raw data, such as start datatime and end
        datatime. ( | is not allowed as it is reserved for col speration)
    Server: server name used if the raw data is queried from database
    Database: database name used if the raw data is queired from database
    Query: sql query string used to get the raw data


Columns in clean_data_info.csv
    RawDataName: name of the raw dataframe
    ColName: name of the column in raw dataframe (eg. \rawdata\data1.csv)
    FileName: name of the cleaned column value, saved in \clean_data\data.csv,
        if None, saved in \rawdata\data.csv
    DtTm: date time marker
    Notes: descriptions

Columns in feature_info.csv
    Name: name of this feature
    HashName: hash code of the name
    Dimension: dimension of this features, eg.N for N level one hotkey encoding
    Notes: description about this feature
    Method: ExtractMethod to get this feature from cleaned data (data cleaning
        does not change dimension or column name)

Columns in extraction_info.csv
    CleanDataName: index of clean data used to extract feature
    FeaturName: name of feature extracted from corresponding clean data

Columns in extractor_info.csv
    Name: name of the extracting method
    CreateDtTm: create date time
    FileName: hashed file name where the backup code is saved, ended with '.py'
        The active code is named after exractor_name.py.
            ''')

    def __init__(self):
        self.initialized = False

        self.name = None
        self.repo_path = None
        self.repo_info_file = None

        self.cleaner_path = None
        self.cleaner_info_file = None
        self.cleaner_set = None

        self.extractor_path = None
        self.extractor_set = None

        self.extraction_info_file = None
        self.extraction_info = None

        self.raw_data_path = {}
        self.raw_data_sql_path = {}
        self.raw_data_info_file = {}
        self.raw_data_log_file = {}

        self.clean_data_path = {}
        self.clean_data_info_file = {}
        self.clean_data_log_file = {}

        self.feature_path = {}
        self.feature_info_file = {}
        self.feature_log_file = {}

        # { 'train': df_raw_data_info}
        self.raw_data_info = {}
        self.clean_data_info = {}
        self.feature_info = {}

        # Loaded data set in array, {'train': [df_raw_data_1, df_raw_data_2]}
        self.raw_data_set = {}
        self.clean_data_set = {}
        self.feature_set = {}

        self.initialized = False
        self._delete_time_mk = None
        self._delete_session_time_mk = None

    @classmethod
    def create_repo_(self):
        print(
            ''' Usage:
            create_repo(self, name, directory=None)
            A repository contains
            \info.dat
            \cleaner
            \extractor
                \hist
                \extractor_info.csv
                \extractor1.py
            \train
                \rawdata\rawdata_info.csv
                \cleanedData\clean_data_info.csv
                \fature\feature_info.csv
                       \feature1\feature1.csv
            \val
            \test
            ''')

    def create_repo(self, name, sessions=['train', 'val', 'test'],
                    directory=None):
        if ((name is None) or (not isinstance(name, str))):
            print("Please provide a project name")
            return
        # set the repository name before initialization so that if ini failed,
        # the use could still delete the repository with repository name
        self.name = name

        if (directory is None):
            print("Directory is not given, creating repository folder " +
                  os.getcwd() + r'\_repo')
            if(os.path.isdir(r'.\_repo')):
                print(
                    'Failed creating: ', os.path.abspath(r'.\_repo') +
                    ''' already exists in current working directory. Please use
                    another path which does not contain any other repositories.
                    If you wnat to load this repository, please use load_repo()
                    ''')
                return
            else:
                directory = os.getcwd()
                path = r'.\_repo'
                os.makedirs(path)
                self.repo_path = os.path.abspath(path)
        elif(not os.path.isdir(directory)):
            print(
                "Please provide a valid directory to save the repository data")
            return
        else:
            path = directory + r'\_repo'
            os.makedirs(path)
            self.repo_path = os.path.abspath(path)

        # Set path
        self.sessions = sessions
        self.set_path(self.repo_path, sessions)

        # initialize files
        try:
            with open(self.repo_info_file, 'a+') as f:
                f.write("Repository Name: %s\n" % name)
                f.write(','.join(sessions) + '\n')

            self._ini_files(sessions)
            self.load_repo(directory)
        except Exception:
            shutil.rmtree(directory)
            raise

        self.initialized = True
        self.session = self.sessions[0]
        print('Repository ' + name + ' is created and initialized.')
        return self

    def add_session(self, s, checkout=False, do_print=True):
        if(self.initialized is False):
            print('Do create_repo() or load_repo() before further operation.')
            return
        if(not isinstance(s, str)):
            raise TypeError('s should be string')
        if(s in self.sessions):
            raise Exception(s + ' exists in current sessions')
        self.sessions.append(s)
        self._add_path_session(self.repo_path, s)
        self._ini_files_session(s)
        self._load_session(s)

        try:
            with open(self.repo_info_file, 'w') as f:
                f.write("Repository Name: %s\n" % self.name)
                f.write(','.join(self.sessions) + '\n')
            self._load_session(s)
        except (FileNotFoundError, PermissionError):
            raise
        if(checkout):
            self.session = s
        if(do_print):
            print(s, ' has been added into repository.')
            print('Current working session is ', self.session)

    @classmethod
    def checkout_(cls):
        print('short name for checkout_session(session_name)')

    def checkout(self, s):
        self.checkout_session(s)

    def checkout_session(self, s):
        if(self.initialized is False):
            print('Do create_repo() or load_repo() before further operation.')
            return
        if((not isinstance(s, str)) or (s not in self.sessions)):
            raise ValueError('Invalid session name')
        self.session = s
        print('Working session:', s)

    @classmethod
    def delete_session_(cls):
        print('''delete_session(session_name, token)''')

    def delete_session(self, s, token=None):
        if(self.initialized is False):
            print('Do create_repo() or load_repo() before further operation.')
            return
        if((not isinstance(s, str)) or (s not in self.sessions)):
            raise ValueError('Invalid session name')
        tokenstr = 'I need delete session'
        if((token is None) or (token != tokenstr)):
            print('Please enter token: ', tokenstr)
            return
        if(self._delete_session_time_mk is None):
            self._delete_session_time_mk = datetime.datetime.utcnow()
            print('Please repeat delete command within 60 seconds')
            return
        else:
            if(
              (datetime.datetime.utcnow() - self._delete_session_time_mk) >
              datetime.timedelta(0, 60, 0)):
                self._delete_session_time_mk = None
                self.delete_session(s, token)
                return
        if(s not in self.sessions):
            raise ValueError('Invalid session name')

        self._unload_session(s)
        self._delete_path_session(s)
        self.sessions.remove(s)
        if(self.repo_path[-5:] != '_repo'):
            raise RuntimeError('Internal error, wrong repo_path')
        try:
            fullpath = self.repo_path + '\\' + s
            shutil.rmtree(fullpath, ignore_errors=False, onerror=None)
        except PermissionError:
            print('Failed to remove ', fullpath)
            raise
        print('Folder and all files in Session ', s, ' has been deleted: ',
              fullpath)

    @classmethod
    def set_path_(self):
        print(
            'Reset the repository folder. Useful when migrate the repository.')

    def set_path(self, repo_path, sessions):
        self.repo_info_file = repo_path + '\\info.dat'

        self.cleaner_path = repo_path + '\\cleaner'
        self.cleaner_hist = self.cleaner_path + '\\hist'
        self.cleaner_info_file = self.cleaner_path + '\\cleaner_info.csv'
        self.cleaner_log_file = self.cleaner_path + '\\cleaner_log.csv'
        if(self.cleaner_path not in sys.path):
            sys.path.insert(0, self.cleaner_path)

        self.extractor_path = repo_path + '\\extractor'
        self.extractorhist_path = self.extractor_path + '\\hist'
        self.extractor_info_file = self.extractor_path + '\\extractor_info.csv'
        self.extractor_log_file = self.extractor_path + '\\extractor_log.csv'
        if(self.extractor_path not in sys.path):
            sys.path.insert(0, self.extractor_path)
        self.extraction_info_file = self.extractor_path + \
            '\\extraction_info.csv'

        for s in sessions:
            self._add_path_session(repo_path, s)

    def _add_path_session(self, repo_path, s):
        self.raw_data_path[s] = repo_path + '\\' + s + '\\rawdata'
        self.raw_data_sql_path[s] = self.raw_data_path[s] + '\\sql'
        self.raw_data_info_file[s] = self.raw_data_path[s] + \
            '\\rawdata_info.csv'
        self.raw_data_log_file[s] = self.raw_data_path[s] + \
            '\\rawdata_log.csv'

        self.clean_data_path[s] = repo_path + '\\' + s + \
            '\\clean_data'
        self.clean_data_info_file[s] = self.clean_data_path[s] + \
            '\\clean_data_info.csv'
        self.clean_data_log_file[s] = self.clean_data_path[s] + \
            '\\clean_data_log.csv'

        self.feature_path[s] = repo_path + '\\' + s + '\\feature'

        self.feature_log_file[s] = self.feature_path[s] + \
            '\\feature_log.csv'
        self.feature_info_file[s] = self.feature_path[s] + \
            '\\feature_info.csv'

    def _delete_path_session(self, s):
        self.raw_data_path.pop(s)
        self.raw_data_sql_path.pop(s)
        self.raw_data_info_file.pop(s)
        self.raw_data_log_file.pop(s)

        self.clean_data_path.pop(s)
        self.clean_data_info_file.pop(s)
        self.clean_data_log_file.pop(s)

        self.feature_path.pop(s)
        self.feature_log_file.pop(s)
        self.feature_info_file.pop(s)

    def _ini_files(self, sessions):
        os.makedirs(self.cleaner_path)
        os.makedirs(self.cleaner_hist)
        with open(self.cleaner_info_file, 'a+') as f:
            f.write('Utc|Name|FileName|Notes')
        with open(self.cleaner_log_file, 'a+') as f:
            f.write('Action|Utc|Name|FileName|Notes')
        self.cleaner_set = []

        os.makedirs(self.extractor_path)
        os.makedirs(self.extractorhist_path)

        with open(self.extractor_info_file, 'a+') as f:
            f.write('Utc|Name|FileName|Notes|TrainedExtractor\n')
        self.extractor_info = self._df_from_csv(self.extractor_info_file)
        with open(self.extractor_log_file, 'a+') as f:
            f.write('Action|Utc|Name|FileName|Notes|TrainedExtractor\n')
        with open(self.extraction_info_file, 'a+') as f:
            f.write('Utc|ExtractorName|CleanDataName\n')
        self.extractor_set = []

        for s in sessions:
            self._ini_files_session(s)

    def _ini_files_session(self, s):
        print(self.raw_data_path[s])
        os.makedirs(self.raw_data_path[s])
        os.makedirs(self.raw_data_sql_path[s])
        with open(self.raw_data_info_file[s], 'a+') as f:
            f.write('Utc|Name|Rows|Cols|Notes|Server|Database|QueryFile\n')
        with open(self.raw_data_log_file[s], 'a+') as f:
            f.write('Action|Utc|Name|Rows|Cols|Notes|Server|' +
                    'Database|QueryFile\n')

        os.makedirs(self.clean_data_path[s])
        with open(self.clean_data_info_file[s], 'a+') as f:
            f.write('Utc|Name|RawDataName|ColName|Notes\n')
        with open(self.clean_data_log_file[s], 'a+') as f:
            f.write('Action|Utc|Name|RawDataName|ColName|Notes\n')

        os.makedirs(self.feature_path[s])
        with open(self.feature_info_file[s], 'a+') as f:
            f.write('Utc|Name|HashName|Dimension|Notes|Method\n')
        with open(self.feature_log_file[s], 'a+') as f:
            f.write('Action|Utc|Name|HashName|Dimension|Notes|Method\n')

    @classmethod
    def load_repo_(self):
        print(''' Loading repository, session=train by default.
        Usage:
        load_repo(self, directory=None)
            directory:  the directory which contains folder \_repo''')

    def load_repo(self, directory=None):
        if (directory is None):
            directory = os.getcwd()
        directory = directory + r'\_repo'
        if(not os.path.isdir(directory)):
            print(
                'Please provide a valid directory that contains repository' +
                ' folder \_repo')
            print('Current directory is: ' + directory)
            return
        self.repo_path = directory
        self.repo_info_file = directory + '\\' + 'info.dat'
        # get repository name and sessions
        if(not os.path.isfile(self.repo_info_file)):
            print("Repository is damaged. info.dat is not found in folder " +
                  self.repo_path)
            return
        with open(self.repo_info_file, 'r') as file:
            ls = file.readlines()

        line = ls[0].lstrip().rstrip('\n')
        if ((len(line) > 1) and (line[0] != "#")):
            self.name = line
        parts = re.split("[,\!?:]+", self.name)
        if ((len(parts) >= 2) and (parts[0].lower() == 'repository name')):
            self.name = parts[1].lstrip().rstrip()
        else:
            print("Repository Name is not found in info.dat")
            return

        self.sessions = ls[1].rstrip('\n').split(',')
        self.set_path(self.repo_path, self.sessions)

        # cleaner_info and cleaner_set
        self.cleaner_info = self._df_from_csv(self.cleaner_info_file)
        if(self.cleaner_info is None):
            print('Failed to load ', self.cleaner_info_file)
            return
        self.cleaner_set = [None] * self.cleaner_info.shape[0]
        # extractor_info and extractor_set
        self.extractor_info = self._df_from_csv(self.extractor_info_file)
        if(self.extractor_info is None):
            print('Failed to load ' + self.extractor_info_file)
            return
        self.extractor_set = [None] * self.extractor_info.shape[0]
        self.extraction_info = self._df_from_csv(
            filepath=self.extraction_info_file)

        # load raw data, clean data, feature info
        for s in self.sessions:
            self._load_session(s)

        self.session = self.sessions[0]
        self.initialized = True
        print("Repository %s is loaded." % self.name)

    def _load_session(self, s):
        print('Loading Session', s)
        self.raw_data_info[s] = self._df_from_csv(
            filepath=self.raw_data_info_file[s])
        self.clean_data_info[s] = self._df_from_csv(
            filepath=self.clean_data_info_file[s])
        self.feature_info[s] = self._df_from_csv(self.feature_info_file[s])

        self.raw_data_set[s] = [None]*(self.raw_data_info[s]).shape[0]
        self.clean_data_set[s] = [None]*(self.clean_data_info[s]).shape[0]
        self.feature_set[s] = [None]*(self.feature_info[s]).shape[0]

    def _unload_session(self, s):
        self.raw_data_info.pop(s)
        self.clean_data_info.pop(s)
        self.feature_info.pop(s)

        self.raw_data_set.pop(s)
        self.clean_data_set.pop(s)
        self.feature_set.pop(s)

    @classmethod
    def delete_repo_(self):
        print('''   Method description
        Remove current repository and delete all the data. Need provide token.
            Usage
        delete_repo(self, name=None, token=None)
        ''')

    def delete_repo(self, name=None, token=None):
        if(self.initialized is False):
            print('Do create_repo() or load_repo() before further operation.')
            return

        tokenstr = 'I am deleting All files in this repository.'
        if(((token is None) or (not isinstance(token, str))) or
           (token != tokenstr)):
            raise AssertionError(
                "Please provide repository name and token: " + tokenstr)

        if(self.name != name):
            self._delete_time_mk = None
            raise ValueError(name + ' is not the correct repository name')

        if(self._delete_time_mk is None):
            self._delete_time_mk = datetime.datetime.utcnow()
            print('You are deleting ', self.repo_path)
            print('Type delete command again within 60 seconds to confirm')
            return
        else:
            if((datetime.datetime.utcnow() - self._delete_time_mk) >
               datetime.timedelta(0, 60, 0)):
                self._delete_time_mk = None
                self.delete_repo(name, token)
                return

        if(self.repo_path[-5:] != '_repo'):
            print('wrong path')
            return
        if (token == tokenstr):
            try:
                shutil.rmtree(
                    self.repo_path, ignore_errors=False, onerror=None)
            except PermissionError:
                print('Failed to remove ' + self.repo_path)
                raise
            print("Folder and all files in the folder has been deleted: " +
                  self.repo_path)

    @classmethod
    def add_raw_data_(self):
        print('''   Usage
        add_raw_data(self, df, name, notes='', over_write=False,
        do_print=True, server='', database='', query='')
            df  : raw data in DataFrame (not matrix because df contains column
                  name)
        ''')

    def add_raw_data(self, df, name, notes='', over_write=False, do_print=True,
                     server='', database='', query='', session=None):
        if(self.initialized is False):
            print('Do create_repo() or load_repo() before further operation.')
            return
        if(session is None):
            session = self.session
        else:
            if(session not in self.sessions):
                raise KeyError('session should be one of ' +
                               ','.join(self.sessions))
        if(do_print):
            print('Working session: ', session)

        raw_data_path = self.raw_data_path[session]
        raw_data_info_file = self.raw_data_info_file[session]
        raw_data_sql_path = self.raw_data_sql_path[session]
        raw_data_log_file = self.raw_data_log_file[session]
        raw_data_info = self.raw_data_info[session]
        raw_data_set = self.raw_data_set[session]

        try:
            self._add_raw_data(
                df, name, notes, over_write, do_print,
                server, database, query, raw_data_info, raw_data_set,
                raw_data_path, raw_data_sql_path, raw_data_info_file,
                raw_data_log_file)
        except Exception:
            print('Failed to add raw data')
            raise

    def _add_raw_data(self, df, name, notes, over_write, do_print, server,
                      database, query, raw_data_info, raw_data_set,
                      raw_data_path, raw_data_sql_path, raw_data_info_file,
                      raw_data_log_file):
        if(not isinstance(df, pd.DataFrame) or df.shape[0] == 0):
            raise TypeError('Invalid data frame provided')
            return
        if((not isinstance(name, str)) or (len(name) < 1) or
           (len(name) > 127)):
            raise ValueError(
                'Pleaes provide nonempty string type name with length<128.')

        if(re.findall('[^A-Za-z0-9-_]', name)):
            raise ValueError('name can should be letters, numbers, - and _',
                             'and start with letters')
            return
        if((notes is not None) and (notes != '')):
            if(not isinstance(notes, str)):
                raise TypeError('Please provide string type notes')
            if(re.findall('[\|]', notes)):
                raise ValueError("notes should not contain |")

        raw_data_full_path = raw_data_path + '\\' + name + '.csv'
        i_raw_data = self._first_index(raw_data_info, 'Name', name)
        if(i_raw_data != -1):
            if (os.path.isfile(raw_data_full_path) and (over_write is False)):
                raise FileExistsError(
                    name + ' exists in current repository' +
                    'If need over write, please set over_write=True')

        # write/overwrite query and raw data to csv file
        # remove old query file when updating raw_date_info
        queryfile = ''
        if((query is not None) and (query != '')):
            hash_object = hashlib.sha256(query.encode('utf-8'))
            queryfile = hash_object.hexdigest() + '.sql'
            try:
                with open(raw_data_sql_path + '\\' + queryfile, 'w+') as f:
                    f.write(query)
            except PermissionError:
                print('Failed to write query to file: ' + queryfile)
                raise

        try:
            df.to_csv(raw_data_full_path, index=False, sep='|')
        except (FileNotFoundError, PermissionError):
            print('Write file error.')
            raise

        # update raw_data_info
        entry = [datetime.datetime.utcnow(), name, df.shape[0], df.shape[1],
                 notes, server, database, queryfile]
        log = [1]
        log.extend(entry)
        try:
            with open(raw_data_log_file, 'a+') as f:
                f.write('|'.join([str(x) for x in log]) + '\n')
        except (FileNotFoundError, PermissionError):
            print('Failed to log data')
            raise
        if (i_raw_data == -1):
            raw_data_set.extend([df])
            raw_data_info.loc[raw_data_info.shape[0]] = entry
            try:
                with open(raw_data_info_file, 'a+') as f:
                    f.write('|'.join([str(x) for x in entry]) + '\n')
            except (FileNotFoundError, PermissionError) as e:
                print('Failed to update ' + raw_data_info_file)
                raise
        else:
            raw_data_set[i_raw_data] = df
            raw_data_info.loc[i_raw_data] = entry
            try:
                raw_data_info.to_csv(
                    raw_data_info_file, index=False, sep='|')
            except PermissionError:
                print('Failed to update ' + raw_data_info_file)
                print('Please try repo.raw_data_info.to_csv ' +
                      '(repo.raw_data_info_file, index=False, sep=\'|\')')
                raise

        if(do_print):
            print(name, df.shape, 'raw data is saved in', raw_data_full_path)

    @classmethod
    def get_raw_data_(self):
        print('''   Usage
        get_raw_data(self, name, reload=False, session=None, do_print=True)
            name:  string type name
            reload: 0=no reload, 1=reload from RAM, 2=reload from file
            session: session name, if None, use current session.
            do_print:   print raw data shape
        ''')

    def get_raw_data(self, name, reload=0, session=None, do_print=True):
        if(self.initialized is False):
            print('Please create_repo() or load_repo() ' +
                  'before any further operation.')
            return None
        # input check
        if((not isinstance(name, str))):
            raise TypeError('Please provide string type name')

        rawdata = None
        print('session= ', session)
        if(session is None):
            session = self.session
        elif (session not in self.sessions):
            raise ValueError('session should be train/val/test only')

        raw_data_info = self.raw_data_info[session]
        raw_data_set = self.raw_data_set[session]
        raw_data_path = self.raw_data_path[session]
        try:
            rawdata = self._get_raw_data(
                name, raw_data_info, raw_data_set, raw_data_path, reload,
                do_print)
        except Exception:
            print('Failed to load ', name)
            raise
        return rawdata

    def _get_raw_data(
      self, name, raw_data_info, raw_data_set, raw_data_path,
      reload, do_print):
        i_raw_data = self._first_index(raw_data_info, 'Name', name)
        if(i_raw_data == -1):
            raise KeyError(name + ' is not found in raw_data_info')

        if(isinstance(raw_data_set[i_raw_data], pd.DataFrame)):
            if((reload == 0) and do_print):
                print(name + ' raw data in memory is returned.')
            elif(reload == 1):
                print(name + ' raw data is reloaded')
            return raw_data_set[i_raw_data]

        fullpath = raw_data_path + '\\' + name + '.csv'
        df = self._df_from_csv(fullpath)
        if(df is None):
            print('Load raw data from ' + fullpath + ' failed.')
            return
        raw_data_set[i_raw_data] = df
        if (do_print):
            print('Loaded raw data from ' + fullpath)
            print(
                'Size: rows=', str(raw_data_set[i_raw_data].shape[0]),
                'cols=' + str(raw_data_set[i_raw_data].shape[1]))
        return raw_data_set[i_raw_data]

    @classmethod
    def get_raw_data_query_(self):
        print('''Get sql query from .sql file in repository.
        Next will add function to get raw data by exec the query.          TODO
        ''')

    def get_raw_data_query(self, name, session=None):
        if(self.initialized is False):
            print(
                'Please create_repo() or load_repo() ' +
                'before any further operation.')
            return None

        if((not isinstance(name, str))):
            print('Please provide string type name')

        if (session is None):
            session = self.session
        elif (session not in ['train', 'val', 'test']):
            raise ValueError('session should be train/val/test only')

        return self._get_raw_data_query(name, session)

    def _get_raw_data_query(self, name, session):
        s = session
        i_raw_data = self._first_index(self.raw_data_info[s], 'Name', name)
        if(i_raw_data == -1):
            print(name + 'is not found in raw_data_info')
            return
        query_file = self.raw_data_info[s].QueryFile[i_raw_data]
        if(query_file is None):
            return None
        fullpath = self.raw_data_sql_path[s] + '\\' + query_file
        if(not os.path.isfile(fullpath)):
            print(fullpath + ' does not exist.')
            return None
        else:
            try:
                with open(fullpath) as f:
                    query = f.read()
            except Exception as e:
                print('Failed to read ' + fullpath)
                raise
        return query

    @classmethod
    def delete_raw_data_(self):
        print('''Method description:
        Delete raw data in order of feature data, clean data, and raw data.
        Ideally all history should be recoreded, but raw data is very large
        most of the time. The user need be able to delete if any changes is
        needed. So only the sql query file is logged.''')

    def delete_raw_data(self, name, session=None, token=None, do_print=True):
        tokenstr = 'I am deleting raw data in repository' + \
                   ' and it is unrecoverable.'
        if (token != tokenstr):
            print('Please enter the token to confirm delete operation. Token:')
            print(tokenstr)
            return

        if(session is None):
            session = self.session
        elif(session not in ['train', 'val', 'test']):
            raise ValueError('session should be train/val/test only')

    def _delete_raw_data(self, name, session, token, do_print):
        s = session
        i_raw_data = self.raw_data_info[s].Name[
            self.raw_data_info[s].Name == name].index
        if(len(i_raw_data) == 0):
            print(name + ' does not exist in current repository.')
            return
        i_raw_data = i_raw_data[0]

        remaining_file = []

        # remove feature
        i_clean_data = self.clean_data_info[s].Name[
            self.clean_data_info[s].RawDataName == name].index
        i_extraction = self.extraction_info.CleanDataName[
            [x in i_clean_data for x in self.extraction_info.CleanDataName]
            ].index
        i_feature = self.extraction_info.FeatureIdx[i_extraction]

        # remove feature files and update feature_info
        if (len(i_feature) > 0):
            for i in i_feature:
                fullpath = self._rm_feature_by_idx(i)
                if(fullpath is not None):
                    remaining_file.extend([fullpath])
            self.feature_info[s] = self.feature_info[s].drop(
                self.feature_info[s].index[i_feature])
            try:
                self.feature_info[s].to_csv(
                    self.feature_info_file[s], index=False, sep='|')
            except PermissionError:
                print('Failed to overwrite ' + self.feature_info_file[s])
                raise

        # update extraction_info
        if(len(i_extraction) > 0):
            self.extraction_info = self.extraction_info.drop(
                self.extraction_info.index[i_extraction])
            try:
                self.extraction_info.to_csv(
                    self.extraction_info_file, index=False, sep='|')
            except PermissionError:
                print('Failed to overwrite ' + self.extraction_info_file)

        # delete clean data and update clean_data_info
        if(len(i_clean_data) > 0):
            for i in i_clean_data:
                fullpath = self._rm_clean_data_by_idx(i)
                if(fullpath is not None):
                    remaining_file.extend([fullpath])
            self.clean_data_info = self.clean_data_info[s].drop(
                self.clean_data_info[s].index[i_clean_data])
            try:
                self.clean_data_info[s].to_csv(
                    self.clean_data_info_file[s], index=False, sep='|')
            except PermissionError:
                print('Failed to overwrite ' + self.clean_data_info_file[s])

        # delete raw data and update raw_data_info
        entry = [0, datetime.datetime.utcnow()]
        record = list(self.raw_data_info.loc[i_raw_data])
        entry.extend(record[1:(len(record)-1)])
        try:
            with open(self.raw_data_log_file[s], 'a+') as f:
                f.write('|'.join([str(x) for x in entry]) + '\n')
        except Exception:
            print('Failed to log deleting raw data action')
            raise

        fullpath = self.raw_data_path[s] + '\\' + name + '.csv'
        if(not os.path.isfile(fullpath)):
            print(fullpath + 'does not exist or it had been deleted.')
        else:
            try:
                os.remove(fullpath)
            except PermissionError:
                print('Failed to remove file: ' + fullpath)
                raise
        self.raw_data_info[s] = self.raw_data_info[s].loc[
            self.raw_data_info[s].index[
                self.raw_data_info[s].index != i_raw_data]]
        self.raw_data_info[s].to_csv(
            self.raw_data_info_file[s], index=False, sep='|')

        print('Files are deleted.')
        if(len(remaining_file) > 0):
            self.remaining_file = remaining_file
            print('Some files are not deleted. Check out repo.remaining_file.')
        return remaining_file

    @classmethod
    def add_clean_data_(self):
        print('''
            Method description
        1d array value is parsed to DataFrame type and saved to csv file.
            Usage
        add_clean_data(self, raw_data_name, col_name, value=None, notes='',
                       over_write=False, session=None, do_print=True)
            value:
                1d array, could be pandas.Series, list, or numpy.ndarray(1d).
                if value=None, will use raw data as clean data, ie no cleaing
                if None, raw_data_name[col_name] is used as value
            session:
                session name, if None, will use current session
        ''')

    def add_clean_data(self, raw_data_name, col_name, value=None, notes='',
                       over_write=False, session=None, do_print=True):
        if (self.initialized is False):
            print('Do create_repo() or load_repo() ' +
                  ' before any further operation.')
        if (session is None):
            session = self.session
        elif(session not in self.sessions):
            print(session, ' should be in ', ','.join(self.sessions))
            return None

        s = session
        raw_data_info = self.raw_data_info[s]
        raw_data_set = self.raw_data_set[s]
        raw_data_path = self.raw_data_path[s]
        clean_data_info = self.clean_data_info[s]
        clean_data_info_file = self.clean_data_info_file[s]
        clean_data_set = self.clean_data_set[s]
        clean_data_path = self.clean_data_path[s]
        clean_data_log_file = self.clean_data_log_file[s]

        self._add_clean_data(
            raw_data_name, col_name, value, notes, over_write, raw_data_info,
            raw_data_set, raw_data_path, clean_data_info, clean_data_info_file,
            clean_data_set, clean_data_path, clean_data_log_file, do_print)

    def _add_clean_data(
      self,
      raw_data_name, col_name, value, notes, over_write, raw_data_info,
      raw_data_set, raw_data_path, clean_data_info, clean_data_info_file,
      clean_data_set, clean_data_path, clean_data_log_file, do_print):
        # sanity check
        i_raw_data = self._first_index(raw_data_info, 'Name',
                                       raw_data_name)
        if(i_raw_data == -1):
            print(raw_data_name + 'is not found in raw_data_info')
            return

        clean_data_name = col_name
        fullpath = clean_data_path + '\\' + clean_data_name + '.csv'

        # If clean data exists in current repo,
        # assume file exists => exists in repo,
        if(os.path.isfile(fullpath)):
            if (over_write is False):
                print(clean_data_name, ' exists in current repository.')
                print('If you need over write please set over_write=True')
                return
            else:
                i_clean_data = self._first_index(clean_data_info, 'Name',
                                                 clean_data_name)
                if(i_clean_data == -1):
                    print('clean data file exists in repository but no logs.')
                    print('Will create new logs and over write old data')
                    over_write = False
                    i_clean_data = clean_data_info.shape[0]
        else:
            over_write = False

        # If raw data is not loaded, load it
        if(raw_data_set[i_raw_data] is None):
            self._get_raw_data(
                raw_data_name, raw_data_info, raw_data_set,
                raw_data_path, True, do_print)
        if(raw_data_set[i_raw_data] is None):
            print('Failed to load raw data: ' + raw_data_name)
            return
        # If col_name not in raw data
        if(col_name not in raw_data_set[i_raw_data].columns):
            print(raw_data_name + 'does not contain column: ' + col_name)
            return

        if(value is None):
            value = raw_data_set[i_raw_data].loc[:, col_name]
            if(sum(pd.isnull(value)) > 0):
                raise ValueError('null exists in value')
        try:
            df = pd.DataFrame({col_name: value})
            df.to_csv(fullpath, index=False, sep='|')
        except Exception:
            print('Write file error.')
            raise

        # Update clean_data_info
        entry = [datetime.datetime.utcnow(), clean_data_name, raw_data_name,
                 col_name, notes]
        if(over_write):
            clean_data_set[i_clean_data] = df
            clean_data_info.loc[i_clean_data] = entry
            log = [2]
        else:
            clean_data_set.extend([df])
            clean_data_info.loc[clean_data_info.shape[0]] = entry
            log = [1]
        log.extend(entry)
        try:
            with open(clean_data_log_file, 'a+') as f:
                f.write('|'.join([str(x) for x in log]) + '\n')
            if(over_write):
                clean_data_info.to_csv(
                    clean_data_info_file, index=False, sep='|')
            else:
                with open(clean_data_info_file, 'a+') as f:
                    f.write('|'.join([str(x) for x in entry]) + '\n')
        except (FileNotFoundError, PermissionError):
            print('Failed to update clean data log or data file')
            self._print_error()
            return

        if(over_write and do_print):
            print('Updated clean data: ', clean_data_name, ' size', df.shape)
        if((not over_write) and do_print):
            print('Added clean data: ', clean_data_name, ' size', df.shape)

    @classmethod
    def get_clean_data_(self):
        print(''' Get clean data as DataFrame
        get_clean_data(self, clean_data_name, do_print=True)
        ''')

    def _get_clean_data(
      self, clean_data_name, clean_data_info, clean_data_path, clean_data_set,
      reload, do_print):
        # list of feature names
        if(isinstance(clean_data_name, pd.Series)):
            clean_data_name = list(clean_data_name)
        if(isinstance(clean_data_name, list)):
            n = len(clean_data_name)
            if(n == 0):
                print('list length should not be 0')
                return None
            idx = [None] * n
            dfs = [None] * n
            for i in range(n):
                name = clean_data_name[i]
                if(not isinstance(name, str) or (len(name) > 255)):
                    print('Please provide name length less than 256')
                    return None
                i_clean_data = self._first_index(clean_data_info, 'Name',
                                                 name)
                if(i_clean_data == -1):
                    print(name + ' is not found in clean_data_info')
                    return None
                idx[i] = i_clean_data

            for i in range(n):
                dfs[i] = self._get_clean_data_by_idx(
                    idx[i], clean_data_info, clean_data_set, clean_data_path,
                    reload, do_print)
            return reduce(lambda left,
                          right: pd.merge(left, right, left_index=True,
                                          right_index=True), dfs)
        # one feature name
        if(not isinstance(clean_data_name, str) or
           (len(clean_data_name) > 255)):
            print('Please provide clean_data_name length less than 256 ')
            return None
        i_clean_data = self._first_index(clean_data_info, 'Name',
                                         clean_data_name)
        if(i_clean_data == -1):
            print(clean_data_name + ' is not found in clean_data_info')
            return None
        return self._get_clean_data_by_idx(
            i_clean_data, clean_data_info, clean_data_set, clean_data_path,
            reload, do_print)

    def get_clean_data(self, clean_data_name, session=None, reload=True,
                       do_print=True):
        if (self.initialized is False):
            print('''Please create_repo() or load_repo() before any further
                  operation.''')
            return None
        if (session is None):
            session = self.session
        elif(session not in self.sessions):
            print(session, ' should be in ', ','.join(self.sessions))
            return None

        s = session
        clean_data_info = self.clean_data_info[s]
        clean_data_set = self.clean_data_set[s]
        clean_data_path = self.clean_data_path[s]

        df = self._get_clean_data(
            clean_data_name, clean_data_info, clean_data_path, clean_data_set,
            reload, do_print)
        return df

    # i_clean_data should be ensured to be valid
    def _get_clean_data_by_idx(
      self, i_clean_data, clean_data_info, clean_data_set, clean_data_path,
      reload, do_print=True):
        df = clean_data_set[i_clean_data]
        clean_data_name = clean_data_info.Name[i_clean_data]
        if(df is not None):
            if(not reload):
                if(do_print):
                    print(clean_data_name + ' was loaded from current memory.')
                return df
        fullpath = clean_data_path + '\\' + clean_data_name + '.csv'
        df = self._df_from_csv(fullpath)
        if(df is None):
            print('Failed to read data in ' + fullpath)
        else:
            clean_data_set[i_clean_data] = df
            if(do_print):
                print('Clean data:', clean_data_name + ' is loaded. Size: ',
                      df.shape)
        return df

    @classmethod
    def delete_clean_data_(self):
        print('delete_clean_data(self, clean_data_name, do_print=True)')

    def delete_clean_data(self, clean_data_name, session=None, do_print=True):
        if (self.initialized is False):
            print('Repo is not initizliaed, Use create_repo() or load_repo()' +
                  ' before any further operation.')
        if(not isinstance(clean_data_name, str)):
            print('Please provide string type clean_data_name')
            return

        if(session is None):
            session = self.session
        elif(session not in self.sessions):
            raise ValueError('session should be ' + ','.join(self.sessions))

        clean_data_path = self.clean_data_path[session]
        clean_data_info = self.clean_data_info[session]
        clean_data_info_file = self.clean_data_info_file[session]
        clean_data_log_file = self.clean_data_log_file[session]
        self._delete_clean_data(
            clean_data_name, clean_data_path, clean_data_info,
            clean_data_info_file, clean_data_log_file, do_print)
        self.clean_data_info[session] = self._df_from_csv(clean_data_info_file)

    def _delete_clean_data(
      self, clean_data_name, clean_data_path, clean_data_info,
      clean_data_info_file, clean_data_log_file, do_print):
        i_clean_data = self._first_index(clean_data_info, 'Name',
                                         clean_data_name)
        if(i_clean_data == -1):
            print(clean_data_name + ' is not found in clean_data_info')
            return

        # backup clean data info row, then update clean_data_info,
        # finally delete clean data file.
        entry = [0, datetime.datetime.utcnow()]
        record = list(clean_data_info.loc[i_clean_data])
        entry.extend(record[1:(len(record)-1)])
        fullpath = clean_data_path + '\\' + clean_data_name + '.csv'

        clean_data_info = clean_data_info[
            clean_data_info.index != i_clean_data]
        try:
            clean_data_info.to_csv(clean_data_info_file, index=False, sep='|')
        except PermissionError:
            print('Failed to update clean_data_info')
            raise

        if(os.path.isfile(fullpath)):
            try:
                os.remove(fullpath)
                with open(clean_data_log_file, 'a+') as f:
                    f.write('|'.join([str(x) for x in entry]) + '\n')
            except PermissionError:
                print('Failed to remove ' + clean_data_name)
                raise
            if(do_print):
                print(clean_data_name + ' is deleted from repository.')
        else:
            if(do_print):
                print(clean_data_name + ' had been deleted before.')

    @classmethod
    def add_extractor_(self):
        print('''Method description:
            Add extractor to repository. Should in session 'train'
            If overwrite==True, current extractor
            will be backed up in a new file with a hashcode as the filename and
            extractor file will be overwritten by new code.
        add_extractor(self, extractor_name, clean_data_name_idx, notes,
            over_write=False, code ='', do_print=True)
        Input:
            extractor_name: name of the extracting method, which should be a
                class inherits from ExtractMethod
            clean_data_name_idx:    list of clean data names or indices in
                clean_data_set[self.sessions[0]], ie the first session,
                usually 'train'
            notes:  should not be empty
            feature_name:   use extractor_name if not provided.
            code:   code in string. If code=None or '', repository will create
                a template file for use to edit. The use needs log_extractor()
                to repository.
        Output:
            If code is provided, return the extractor (a class inherits from
            ExtractMethod) imported from the file just saved.
            If code is not provided, return string type fullpath''')

    @classmethod
    def train_extractor_(cls):
        print(''' Train extractor by data from clean_data
            Usage:
        train_extractor(self, extractor_name, df=None, do_print=True)
            df: training data. If not given, will use clean_data_name in
                extraction_info
        ''')

    def train_extractor(self, extractor_name, df=None, do_print=True):
        extr = self.get_extractor(extractor_name)
        if(df is None):
            clean_data_name = \
                self.extraction_info[
                    self.extraction_info.ExtractorName == extractor_name]\
                .CleanDataName
            df = self.get_clean_data(clean_data_name)
        extr.train(df)
        # pickle data to file.
        hash_object = hashlib.sha256(
            (extractor_name + str(datetime.datetime.utcnow())).
            encode('utf-8'))
        modelfile = hash_object.hexdigest() + '.pk'
        pickle.dump(extr,
                    open(self.extractorhist_path + '\\' + modelfile, 'wb+'))
        # save file set_path
        i_extractor = self.extractor_info[
            self.extractor_info.Name == extractor_name].index[0]
        filename = self.extractor_info.loc[i_extractor, 'FileName']
        notes = self.extractor_info.loc[i_extractor, 'Notes']
        self._update_extractor_log(i_extractor, extractor_name, filename,
                                   notes, modelfile, True, do_print)

    def add_extractor(
      self, extractor_name, clean_data_name_idx, notes, feature_name=None,
      over_write=False, code='', do_print=True):
        if(self.session != 'train'):
            print('Please set session to train.')
            return
        if(self.initialized is False):
            print('Please create_repo() or load_repo() ' +
                  'before any further operation.')
            return

        # i_extractor is used in insert/updating self.extractor_info.
        # extractor exists in repo && over_write    ==> over write
        # extractor not exists     && over_write    ==> new extractor
        # extractor exists in repo && not over_write => stop
        # extractor not exists     && not over_write => new extractor
        i_extractor = self._first_index(self.extractor_info, 'Name',
                                        extractor_name)
        if(i_extractor != -1):
            if(not over_write):
                print(extractor_name, ' exists in extractor_info.')
                return
            elif (do_print):
                print(extractor_name, ' will be overwritten.')
        else:
            over_write = False
            i_extractor = self.extractor_info.shape[0]

        # check clean_data_name_idx, replace index (int) with name (str)
        if(not isinstance(clean_data_name_idx, list)):
            print('clean_data_name_idx should be a list of names or index')
            return None
        s = self.sessions[0]
        clean_data_info = self.clean_data_info[s]
        for i in range(len(clean_data_name_idx)):
            item = clean_data_name_idx[i]
            if(isinstance(item, int)):
                if((item >= clean_data_info.shape[0]) or (item < 0)):
                    print(clean_data_name_idx[i], ' out of range')
                    return
                else:
                    clean_data_name_idx[i] = clean_data_info.Name[item]
            elif(isinstance(item, str)):
                i_clean_data = self._first_index(clean_data_info, 'Name',
                                                 item)
                if(i_clean_data == -1):
                    print(item, ' is not found in clean_data_info')
                    return
            else:
                print(item, ' is not int or string')
                return
        # If code is provided, backup code file, save code file as active file
        # If not provided, create template for use to edit, and
        # code needs to be backed when update/delete occurs
        filename = None
        needbackup = True
        if((code is None) or (code == '')):
            needbackup = False
            code = '''
import numpy as np
class MyExtractor:
    def __ini__(self):
        self.trained = False

    def train(self, cleandata):
        self.trained = True

    def extract(self, cleandata):
        if(self.trained is False):
            raise RuntimeError('Please train() before predict()')
        feature=np.concatenate(cleandata, axis=1)
        # put your code here
        return feature
            '''
        # always write code (template or in use) to file for active extractor
        fullpath = self.extractor_path + '\\' + extractor_name + '.py'
        try:
            with open(fullpath, 'w+') as f:
                f.write(code)
        except PermissionError:
            print('Failed to write code file: ', extractor_name)
            raise
        if(not needbackup):
            print(fullpath, ' is generated as a template.')
        else:
            print(fullpath, ' is saved.')

        if(needbackup):
            hash_object = hashlib.sha256(
                (extractor_name + str(datetime.datetime.utcnow())).
                encode('utf-8'))
            filename = hash_object.hexdigest() + '.py'
            backuppath = self.extractorhist_path + '\\' + filename
            try:
                with open(backuppath, 'w+') as f:
                    f.write(code)
            except PermissionError:
                print('Failed to write logging code ', backuppath)
                self._print_error
                return None

        # reload the module
        try:
            extractor = __import__(extractor_name)
            if(over_write):
                extractor = imp.reload(sys.modules[extractor_name])
        except Exception:
            print('Failed to load module from ' + fullpath)
            self._print_error()
            return None

        # update extractor_info file and extractor_log_file
        self._update_extractor_log(i_extractor, extractor_name, filename,
                                   notes, None, over_write, do_print)
        # update extraction_info
        self._update_extraction_file(extractor_name, clean_data_name_idx)
        # update extractor_set in running environment
        ex = extractor.MyExtractor()
        if(i_extractor < len(self.extractor_set)):
            self.extractor_set[i_extractor] = ex
        else:
            self.extractor_set.append(ex)
        return ex

    def _update_extractor_log(self, i_extractor, extractor_name, filename,
                              notes, trainedmodel, over_write, do_print):
        entry = [datetime.datetime.utcnow(), extractor_name,
                 filename, notes, trainedmodel]
        self.extractor_info.loc[i_extractor] = entry
        log = '1|' + '|'.join([str(x) for x in entry]) + '\n'
        try:
            self.extractor_info.to_csv(self.extractor_info_file, index=False,
                                       sep='|')
            with open(self.extractor_log_file, 'a+') as f:
                f.write(log)
        except Exception:
            print('Failed to save extractor_info to file or updating log file')
            self._print_error()
            return None
        if(do_print):
            if (over_write):
                print(extractor_name, ' has been updated')
            else:
                print(extractor_name, ' has been added to repository')

    def _update_extraction_file(self, extractor_name, clean_data_name):
        self.extraction_info = self.extraction_info[
            self.extraction_info.ExtractorName != extractor_name] \
            .copy().reset_index(drop=True)
        for name in clean_data_name:
            entry = [str(datetime.datetime.utcnow()), extractor_name, name]
            self.extraction_info.loc[self.extraction_info.shape[0], :] = entry
            # print('added', entry)
        try:
            self.extraction_info.to_csv(
                self.extraction_info_file, index=False, sep='|')
        except Exception:
            print('Failed to update extraction_info file')
            return None

    @classmethod
    def get_extractor_(cls):
        print('''
            Get extractor, a class inherits from ExtractMethod, which contains
            a method extract()
                Usage
            get_extractor(self, extractor_name)
        ''')

    def get_extractor(self, extractor_name):
        if(self.initialized is False):
            print('Please create_repo() or load_repo() ' +
                  'before any further operation.')
            return
        i_extractor = self._first_index(self.extractor_info, 'Name',
                                        extractor_name)
        if(i_extractor == -1):
            print(extractor_name, ' is not found in extractor_info')
            return
        return self._get_extractor_by_idx(i_extractor)

    def _get_extractor_by_idx(self, i_extractor):
        extractor = __import__(self.extractor_info.Name[i_extractor])
        extractor = extractor.MyExtractor()
        trained_extractor = self.extractor_info.TrainedExtractor[i_extractor]
        if(not pd.isnull(trained_extractor)):
            extractor = pickle.load(
                open(self.extractorhist_path + '\\' + trained_extractor, 'rb'))
        return extractor

    @classmethod
    def get_extractor_str_(cls):
        print('''Get extracting method code.
            Usage
        get_extractor_str(self, extractor_name)''')

    def get_extractor_str(self, extractor_name, MAX_SIZE=10000):
        if(self.initialized is False):
            print('Please create_repo() or load_repo() ' +
                  'before any further operation.')
            return
        i_extractor = self._first_index(self.extractor_info, 'Name',
                                        extractor_name)
        if(i_extractor == -1):
            print(extractor_name, ' is not found in extractor_info')
            return
        fullpath = self.extractor_path + '\\' + \
            self.extractor_info.Name[i_extractor] + '.py'
        size = os.path.getsize(fullpath)
        size = max(MAX_SIZE, size)
        code = ''
        try:
            with open(fullpath) as f:
                code = f.read(size)
        except (FileNotFoundError, PermissionError):
            print('Failed to read file: ' + fullpath)
            raise
        return code

    @classmethod
    def log_extractor_(cls):
        print(''' Method description:
        Update existing extractor, which will over write the code.
        User could add_extractor providing None as code.
            Usage:
        log_extractor(extractor_name)
        ''')

    def log_extractor(self, extractor_name):
        if(self.initialized is False):
            print('Please create_repo() or load_repo() ' +
                  'before any further operation.')
            return
        i_extractor = self._first_index(self.extractor_info, 'Name',
                                        extractor_name)
        if(i_extractor == -1):
            print(extractor_name, ' is not found in extractor_info')
            return
        return self._log_extractor(i_extractor)

    ''' Method description:
        Update extractor_info to the latest version and log it.
    '''
    def _log_extractor(self, i_extractor, do_print=True):
        if(i_extractor >= self.extractor_info.shape[0]):
            print('index of extractor_info out of bound')
            return None
        extractor_name = self.extractor_info.Name[i_extractor]
        dttm = datetime.datetime.utcnow()

        fullpath = self.extractor_path + '\\' + extractor_name + '.py'
        hash_object = hashlib.sha256((extractor_name + str(dttm)).
                                     encode('utf-8'))
        filename_hist = hash_object.hexdigest() + '.py'
        if(not os.path.isfile(fullpath)):
            print(fullpath + 'could not be found')
            return None

        # user may or may not load extractor after create/load repository
        try:
            extractor = __import__(extractor_name)
            extractor = imp.reload(sys.modules[extractor_name])
        except Exception:
            print('Failed to import or reload extractor')
            raise

        shutil.copy(fullpath, self.extractorhist_path + '\\' + filename_hist)
        self.extractor_info.loc[i_extractor, 'FileName'] = filename_hist
        entry = [str(x) for x in self.extractor_info.loc[i_extractor]]
        entry = '3|' + '|'.join(entry[1::]) + '\n'
        try:
            with open(self.extractor_log_file, 'a+') as f:
                f.write(entry)
        except (FileNotFoundError, PermissionError):
            print('Failed to log file from', fullpath, ' to ', filename_hist)
            return None

        self.extractor_info.Utc[i_extractor] = dttm
        self.extractor_info.FileName[i_extractor] = filename_hist
        try:
            self.extractor_info.to_csv(self.extractor_info_file,
                                       index=False, sep='|')
        except (PermissionError):
            print('Failed to update ', self.extractor_info_file)
            raise
        if(do_print):
            print(entry)
        return extractor.extract

    @classmethod
    def delete_extractor_(cls):
        print('delete_extractor(self, extractor_name, do_print=True)')

    def delete_extractor(self, extractor_name, do_print=True):
        if(self.initialized is False):
            print('Please create_repo() or load_repo() ' +
                  'before any further operation.')
            return
        i_extractor = self.extractor_info.Name[
            self.extractor_info.Name == extractor_name].index
        if(len(i_extractor) == 0):
            print(extractor_name, ' is not found in extractor_info')
            return None

        remaining_file = []
        # update extractor_info before updat log file.
        for i in i_extractor:
            fullpath = self.extractor_path + '\\' + \
                self.extractor_info.Name[i] + '.py'
            if(os.path.isfile(fullpath)):
                if(self.extractor_info.FileName[i] is None):
                    self._log_extractor(i)
                try:
                    os.remove(fullpath)
                except PermissionError:
                    print('Failed to delete ', fullpath)
                    remaining_file.extend([fullpath])
                    raise
            elif(do_print):
                print(extractor_name, ' was deleted sometime before.')

        extractor_info_drop = self.extractor_info.loc[i_extractor].copy()
        self.extractor_info = self.extractor_info.drop(i_extractor). \
            reset_index(drop=True)
        try:
            self.extractor_info.to_csv(self.extractor_info_file, index=False,
                                       sep='|')
        except PermissionError:
            print('Failed to update ', self.extractor_info_file)
            return
        try:
            with open(self.extractor_log_file, 'a+') as f:
                for i in extractor_info_drop.index:
                    log = [0, datetime.datetime.utcnow()]
                    entry = extractor_info_drop.loc[i]
                    # in deleting, note is not logged.
                    log.extend(entry[1:(len(entry)-1)])
                    f.write('|'.join([str(x) for x in log]) + '|\n')
        except PermissionError:
            print('Failed to update ', self.extractor_log_file)
            self._print_error()
            return None
        if(do_print):
            print(extractor_name, ' is deleted,', len(remaining_file),
                  ' files not deleted and are returned')
        return remaining_file

    @classmethod
    def add_feature_(cls):
        print(''' Method description
        This method extracts feature from one or more clean data (a column of a
        DataFrame) into one_feature, which contains one or more column vectors
        and saved the data. Note: if extractor_name is provided, feature_values
        should also be provided.
            Usage
        add_feature(feature_name, feature_values=None, feature_name_dtl=None,
        extractor_name=None, notes='', over_write=False, session=None,
        do_print=True)

        feature_name    : name of this feature, should be unique
        clean_data_name : list of clean data names which are used in feature
                          extraction
        method          : class inherited from ExtractMethod, which inherits
                          or override static help(), static extract(),
                          static name_dtl() method. It is essentially a wrap up
                          of extracting operation, even it reuses the same
                          class eg RandomForest) but with different
                          parameeters.''')

    def add_feature(
      self, feature_name, feature_values=None, feature_name_dtl=None,
      extractor_name=None, notes='', over_write=False, session=None,
      do_print=True):
        if(self.initialized is False):
            print('Please create_repo() or load_repo() ' +
                  'before any further operation.')
            return
        if(session is None):
            session = self.session
        elif(session not in self.sessions):
            print(session, ' is not in ', ', '.join(self.sessions))
            return

        if((extractor_name is not None) and
           (not isinstance(extractor_name, str))):
            print('extractor_name should be string')
            return None
        if(session is None):
            session = self.session
        elif(session not in self.sessions):
            print('session should be in ', ', '.join(self.sessions))
            return None

        return self._add_feature(
            feature_name, feature_values, feature_name_dtl, session,
            extractor_name, notes, over_write, do_print)

    '''
    Input:
        i_feature: index of existing feature, otherwise -1.
    '''
    def _add_feature(
      self, feature_name, feature_values, feature_name_dtl, session,
      extractor_name, notes, over_write, do_print):
        s = session
        clean_data_info = self.clean_data_info[s]
        extraction_info = self.extraction_info
        feature_path = self.feature_path[s]
        feature_info_file = self.feature_info_file[s]
        feature_info = self.feature_info[s]
        feature_set = self.feature_set[s]

        i_feature = self._first_index(feature_info, 'Name', feature_name)
        if ((i_feature != -1) and (not over_write)):
            print(
                feature_name +
                ' exists in current feature_info, please use another name')
            return None
        # get clean_data_name
        if(extractor_name is None):
            clean_data_name = [feature_name]
            print('extractor_name is not provided, use clean_data with name: ',
                  feature_name)
        else:
            i_extractor = self._first_index(self.extractor_info, 'Name',
                                            extractor_name)
            if(i_extractor == -1):
                print(extractor_name, ' does not exist in extractor_info')
                return None
            clean_data_name = extraction_info.CleanDataName[
                extraction_info.ExtractorName == extractor_name]
        # check clean_data_name
        clean_data_idx = [self._first_index(clean_data_info, 'Name', x)
                          for x in clean_data_name]
        if(sum([x == -1 for x in clean_data_idx]) > 0):
            print(clean_data_name[clean_data_idx == -1],
                  ' not found in clean_data_info')
            return None
        # check feature_values
        if(feature_values is None):
            if(extractor_name is not None):
                print('If extractor_name is provided, feature_values must',
                      ' be provided as well')
                return None
            else:
                print('feature_values not provided, trying use clean_data',
                      ' with the same name')
                try:
                    feature_values = self.get_clean_data(feature_name,
                                                         session=s)
                except Exception:
                    raise
        if(isinstance(feature_values, list) or
           isinstance(feature_values, pd.Series)):
            feature_values = np.reshape(feature_values, (-1, 1))

        # check feature_name_dtl
        n_values = feature_values.shape[1]
        if(feature_name_dtl is None):
            n_values = feature_values.shape[1]
            if(n_values == 1):
                feature_name_dtl = [feature_name]
            else:
                feature_name_dtl = [feature_name + str(i) for i in range(
                    n_values)]
        elif(len(feature_name_dtl) != n_values):
            print('Length of feature_name_dtl does nto match feature_values')
            return None

        # add feature value, change bool to int
        df = pd.DataFrame(feature_values)
        types = df.apply(np.dtype)
        booltype = types[types == 'bool'].index
        for icol in booltype:
            df.loc[:, icol] = [int(x) for x in df.loc[:, icol]]
        df.columns = feature_name_dtl
        fullpath = feature_path + '\\' + feature_name + '.csv'
        try:
            df.to_csv(fullpath, index=False, sep='|')
        except PermissionError:
            print('Failed to save feature data to ' + fullpath)
            raise

        if(i_feature != -1):
            feature_set[i_feature] = df
        else:
            feature_set.extend([df])
            i_feature = len(feature_set) - 1

        # update feature_info
        dttm = datetime.datetime.utcnow()
        hash_object = \
            hashlib.sha256((str(dttm) + feature_name).encode('utf-8'))
        hashname = hash_object.hexdigest() + '.sql'
        entry = [dttm, feature_name, hashname, feature_values.shape[1], notes,
                 None]

        entry[len(entry)-1] = extractor_name
        feature_info.loc[i_feature] = entry
        try:
            feature_info.to_csv(feature_info_file, index=False, sep='|')
        except PermissionError:
            print('Failed to update featuer_info file')
            return None

        return feature_set[i_feature]

    @classmethod
    def extract_feature_(cls):
        print(''' Method description
        Extract feature from clean data.
            Usage
        extract_feature(feature_name, extractor_name_idx=None,
            notes='', feature_name_dtl=None, session=None, over_write=False,
            do_print=True)
        feature_name    : string
        extractor_name_idx  : extractor name or index in extractor_info. If
            None, will use clean_data with the same name as feature_name
            directly.
        feature_name_dtl    : names of each column in extracted feature. Length
                              must match the feature. If not provided, will
                              use feature_name_Id, Id is sequencial number.
        ''')

    def extract_feature(
      self, feature_name, extractor_name_idx=None,
      notes='', feature_name_dtl=None, session=None, over_write=False,
      do_print=True):
        if(self.initialized is False):
            print('Please create_repo() or load_repo() ' +
                  'before any further operation.')
            return
        if(session is None):
            session = self.session
            print('session: ', self.session)
        elif(session not in self.sessions):
            print('session should be in ', ', '.join(self.sessions))
            return None

        return self._extract_feature(
            feature_name, extractor_name_idx, notes, feature_name_dtl,
            session, over_write, do_print)

    def _extract_feature(
      self, feature_name, extractor_name_idx, notes, feature_name_dtl,
      session,
      over_write, do_print):
        s = session
        extraction_info = self.extraction_info
        feature_info = self.feature_info[s]

        # if overwrite and i_feature >= 0 then overwrite. otherwise -1.
        i_feature = self._first_index(feature_info, 'Name', feature_name)
        if ((i_feature != -1) and (not over_write)):
            print(
                feature_name +
                ' exists in current feature_info, please use another name')
            return None

        # get extractor and set extractor name as string
        if (extractor_name_idx is None):
            print('extractor_name not provided, will use clean_data directly')
            return self.add_feature(
                feature_name, over_write=over_write, do_print=do_print)
        elif(isinstance(extractor_name_idx, str)):
            i_extractor = self._first_index(self.extractor_info, 'Name',
                                            extractor_name_idx)
            if(i_extractor != -1):
                extractor_name = self.extractor_info.Name[i_extractor]
                extractor = self.get_extractor(extractor_name)
            else:
                raise IndexError(str(extractor_name_idx) +
                                 ' is not found in extractor_info')
        elif (isinstance(extractor_name_idx, int)):
            if((extractor_name_idx >= self.extractor_info.shape[0]) or
               (extractor_name_idx < 0)):
                raise IndexError(extractor_name_idx, ' out of bound')
            else:
                extractor_name = self.extractor_info.Name[extractor_name_idx]
                extractor = self._get_extractor_by_idx(extractor_name_idx)

        # get clean data
        try:
            clean_data_name = extraction_info.CleanDataName[
                extraction_info.ExtractorName == extractor_name]
            df = self.get_clean_data(list(clean_data_name), session=s)
        except Exception:
            print('Failed to get clean_data: ', extractor_name)
            raise
        print('clean data sahpe: ', df.shape)
        # extract feature, save it in DataFrame then to csv file.
        feature_values = extractor.predict(df)
        print(type(feature_values))

        return self._add_feature(
            feature_name, feature_values, feature_name_dtl, session,
            extractor_name, notes, over_write, do_print)

    @classmethod
    def get_feature_(cls):
        print(''' Get feature in DataFrame type.
        get_feature(self, feature_name, session=None, reload=False)
        ''')

    def _feature_name_id_check(self, feature_name_or_id, session):
        if(isinstance(feature_name_or_id, list)):
            idlist = []
            namelist = []
            for item in feature_name_or_id:
                a, b = self._feature_name_id_check(item, session)
                idlist.append(a)
                namelist.append(b)
            return idlist, namelist
        if(isinstance(feature_name_or_id, int)):
            if(feature_name_or_id >= self.feature_info[session].shape[0]):
                print('feature index', feature_name_or_id, 'out of boundary')
                return None, None
            feature_name = self.feature_info[session].Name[feature_name_or_id]
            return feature_name_or_id, feature_name
        elif(isinstance(feature_name_or_id, str)):
            feature_id = self._first_index(self.feature_info[session],
                                           'Name', feature_name_or_id)
            if(feature_id == -1):
                print('Feature not found:', feature_name_or_id)
                return None, None
            else:
                return feature_id, feature_name_or_id
        else:
            print('Pleae provide feature name as str or id as int')
            return None, None

    def get_feature(self, feature_name_or_id, session=None, reload=False):
        if (self.initialized is False):
            print('Please create_repo() or load_repo() ' +
                  ' before any further operation.')
            return None
        if(session is None):
            session = self.session
        elif(session not in self.sessions):
            print('session should be in ', ', '.join(self.sessions))

        feature_id, feature_name = self._feature_name_id_check(
            feature_name_or_id, session)
        if(feature_id is None):
            return
        if(isinstance(feature_id, list)):
            check = [x is None for x in feature_id]
            if(any(check)):
                print(feature_id[check.index(None)], 'is not found')
                return

        return self._get_feature(feature_name, session, reload)

    def _get_feature(self, feature_name, session, reload):
        s = session
        feature_path = self.feature_path[s]
        feature_info = self.feature_info[s]
        feature_set = self.feature_set[s]

        # list of feature names
        if(isinstance(feature_name, list)):
            i_feature = [self._first_index(feature_info, 'Name', x)
                         for x in feature_name]
            invalid = [x == -1 for x in i_feature]
            if(sum([int(x) for x in invalid]) > 0):
                print(feature_name[invalid], ' not exist in feature_info')
                return None
            dfs = [self.get_feature(x, session) for x in feature_name]
            return reduce(
                lambda left, right: pd.merge(
                    left, right, left_index=True, right_index=True), dfs)

        if(not isinstance(feature_name, str)):
            print('Please provide feature_name as string')
            return None

        i_feature = self._first_index(feature_info, 'Name', feature_name)
        if(i_feature == -1):
            print(feature_name + ' is not found in feature_info')
            return None
        df = feature_set[i_feature]
        if((df is not None) and (not reload)):
            return df

        fullpath = feature_path + '\\' + feature_name + '.csv'
        df = self._df_from_csv(fullpath)
        if (df is None):
            print('Failed to load file ' + fullpath)
            return None
        feature_set[i_feature] = df
        return df

    @classmethod
    def delete_feature_(cls):
        print('delete_feature(self, feature_name, do_print= True)')

    def delete_feature(self, feature_name, session=None, do_print=True):
        if (self.initialized is False):
            print('Please create_repo() or load_repo() ' +
                  ' before any further operation.')
            return None
        if(session is None):
            session = self.session
        elif(session not in self.sessions):
            print('session should be one of the sessions')
            return
        s = session
        feature_info = self.feature_info[s]
        feature_info_file = self.feature_info_file[s]
        feature_path = self.feature_path[s]

        remaining_files = []
        i_feature = self._first_index(feature_info, 'Name', feature_name)
        if(i_feature == -1):
            print(feature_name + ' is not found in feature_info')
            return remaining_files
        feature_info = feature_info.loc[feature_info.Name != feature_name]
        try:
            feature_info.to_csv(feature_info_file, index=False, sep='|')
            self.feature_info[s] = feature_info
        except PermissionError:
            print('Failed to update feature_info')
            raise
        fullpath = feature_path + '\\' + feature_name + '.csv'
        if(os.path.isfile(fullpath)):
            try:
                os.remove(fullpath)
            except PermissionError:
                print('Failed to remove ', fullpath)
                remaining_files = [fullpath]
                raise
            print(feature_name, 'has been deleted.')
        else:
            print(fullpath, 'not found, it might be deleted manually.')
        return remaining_files

    @classmethod
    def filter_(cls):
        print('''filter(self, boundary): boundary should be list with Length
        of 2 * (feature number), each feature has left (exclusive) and
        right (inclusive) boundary
        return: np.array of booltype
        ''')

    def filter(self, boundary, left_included=True, right_included=False,
               session=None):
        if(session is None):
            session = self.session
        elif(session not in self.sessions):
            print('session should be one of', self.sessions)

        if(not isinstance(boundary, list) or len(boundary) % 2 != 0 or
           len(boundary) > self.feature_repo[session].shape[0] or
           any([isinstance(x, numbers.Numer) for x in boundary])):
            print('Please provide boundary as a list, see:')
            self.filter_()
        f = np.array([True] * (self.get_feature(0, session).shape[0]))
        for i in range(len(boundary)/2):
            if(boundary[i*2] is not None):
                if(left_included):
                    f = f & np.array(self.get_feature(
                        i, session).iloc[:, 1] >= boundary[i])
                else:
                    f = f & np.array(self.get_feature(
                        i, session).iloc[:, 1] > boundary[i])
            if(boundary[i*2 + 1] is not None):
                if(right_included):
                    f = f & np.array(self.get_feature(
                        i, session).iloc[:, 1] <= boundary[i])
                else:
                    f = f & np.array(self.get_feature(
                        i, session).iloc[:, 1] < boundary[i])
        return f

        def filter_single(
          self, feature_name_or_id, left_boundary, right_boundary,
          left_included=True, right_included=False, session=None):
            if(session is None):
                session = self.session
            elif(session not in self.sessions):
                print('session should be one of', self.sessions)

            feature_id, feature_name = self._feature_name_id_check(
                feature_name_or_id, session)
            if(feature_id is None):
                return
            feature = self.get_feature(feature_id, session)
            f = np.array([True]*feature.shape[0])
            if(left_boundary is not None):
                if(left_included):
                    f = f & np.array(feature.iloc[:, 1] >= left_boundary)
                else:
                    f = f & np.array(feature.iloc[:, 1] > left_boundary)
            if(right_boundary is not None):
                if(right_included):
                    f = f & np.array(feature.iloc[:, 1] <= right_boundary)
                else:
                    f = f & np.array(feature.iloc[:, 1] < right_boundary)
    '''---------------------------------------------------------------------'''
    ''' Other Private Methods'''
    '''---------------------------------------------------------------------'''

    ''' Method description
        Get the first index of matched value in df.column
    '''
    def _first_index(self, df, colname, value):
        idx = df.loc[:, colname][df.loc[:, colname] == value].index
        if(len(idx) == 0):
            return -1
        else:
            return idx[0]

    '''
    Method description:
        Delete clean data by index in clean_data_info. Notice that
        clean_data_info is not updated. It will be updated after deleting.
    Output:
        None if succeeded, file path if failed.
    '''
    def _rm_clean_data_by_idx(self, idx):
        fullpath = self.clean_data_path + '\\' + \
            self.clean_data_info.Name[idx] + '.csv'
        if(os.path.isfile(fullpath)):
            try:
                os.remove(fullpath)
            except PermissionError:
                return fullpath
        return None

    '''
    Input:
        list of index or names of clean data
    Return:
        clean data index
    '''
    def _get_clean_data_idx(self, clean_data_name_or_idx):
        if (self.initialized is False):
            print('Please create_repo() or load_repo()' +
                  ' before any further operation.')
            return None
        # if clean_data_name_or_idx is list of names, convert it to index.
        if(not isinstance(clean_data_name_or_idx, list)):
            print('Please provide clean_data_name_or_idx in type of list')
            return None
        clean_data_idx = [None]*len(clean_data_name_or_idx)

        for i in range(len(clean_data_name_or_idx)):
            name = clean_data_name_or_idx[i]
            if(isinstance(name, str)):
                i_clean_data = self.clean_data_info. \
                    Name[self.clean_data_info.Name == name].index
                if(len(i_clean_data) == 0):
                    print(name + ' is not in clean_data_info.')
                    return None
                else:
                    clean_data_idx[i] = i_clean_data[0]
            elif (isinstance(name, int)):
                if((name >= 0) and (name < len(self.clean_data_set))):
                    clean_data_idx[i] = clean_data_name_or_idx[i]
                else:
                    print(clean_data_name_or_idx[i], ' out of range')
                    return None

        return clean_data_idx

    '''
    Method description:
        Delete feature file by index in feature_info. Notice that feature_info
        is not updated. It will be updated after deleting.
    Output:
        None if succeeded, file path if failed.
    '''
    def _rm_feature_by_idx(self, idx):
        fullpath = self.feature_path + '\\' + \
            self.feature_info.HashName[idx] + '.csv'
        if(os.path.isfile(fullpath)):
            try:
                os.remove(fullpath)
            except PermissionError:
                print(fullpath)
                raise
        return None

    '''
    Description: a safe way to read csv to pandas.DataFrame. index=False,
        sep='|'
    Output: DataFrame of read success, None if failed.
    '''
    def _df_from_csv(self, filepath):
        df = None
        if (not os.path.isfile(filepath)):
            info = filepath + ' does not exist. Repository is damaged.'
            raise FileNotFoundError(info)
        try:
            df = pd.read_csv(filepath, sep='|')
        except (FileNotFoundError, PermissionError):
            info = filepath + ' is damaged.'
            raise RuntimeError(info)
        return df

    def _print_error(self):
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, exc_obj, fname, exc_tb.tb_lineno)

    ''' Not used yet'''
    def _to_col_vector(self, data):
        if(isinstance(data, np.ndarray)):
            if(len(data.shape) > 2):
                print('Dimension > 2 is not allowed')
                return
            else:
                return data.reshape(-1, 1)
        elif(isinstance(data, list)):
            return np.array(data).reshape(-1, 1)
        elif(isinstance(data, pd.Series)):
            return data.values.reshape(-1, 1)
        else:
            return None
