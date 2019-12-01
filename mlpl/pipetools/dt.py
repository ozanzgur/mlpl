import pandas as pd
import scipy
import numpy as np
from .. import prep
import pickle
import os

class DTIndexer():
    # Provides iloc functionality
    
    def __init__(self, dt):
        # Get datatable
        self.dt = dt
        
    def __getitem__(self, idx):
        selected_data = self.dt.data()
        if isinstance(selected_data, pd.DataFrame):
            return selected_data.iloc[idx]
        else:
            return selected_dat[idx]

class DataTable():
    def __init__(self, data = None, sp = None):
        self.df = pd.DataFrame()
        self.sp = dict()
        self.columns = pd.Index([])
        self.iloc = DTIndexer(self)
        
        # Get dataframe
        if data is not None:
            if isinstance(data, pd.DataFrame):
                self.df = data
                self.columns = data.columns

                # sp is a dict if no argument was passed.
                if sp is not None:
                    if not isinstance(sp, dict):
                        raise TypeError('sp must be a dictionary.')
                    self.sp = sp
                    self.columns = self.columns.append(pd.Index(sp.keys()))
                    self.iloc = DTIndexer(self)

            # Load from file if a filename was given
            elif isinstance(data, str):
                self.load(data)
            else:
                raise TypeError('"data" must be either a dataframe or a filename.')
            
    def dtypes(self):
        types = dict()
        if len(self.df.columns) > 0:
            # Add df columns as they are
            types.update(self.df.dtypes)
            
        if self.sp is not None:
            # Add sparse columns as sparse
            types.update({col: 'sparse' for col in self.sp.keys()})
        return pd.Series(types)
    
    def has_sparse(self):
        return len(self.sp) > 0
    
    def __getitem__(self, col):
        """** DOES NOT RETURN OBJECT COLUMNS **
        
        """
        if isinstance(col, list) or isinstance(col, pd.Index) or isinstance(col, set):
            self.check_all_str(col)
            
            # Get dataframe columns and sparse columns
            df_cols, sp_cols = self.get_cols_by_type(col)
            
            if len(df_cols) == 0 and len(sp_cols) > 0:
                # Take data from only sparse part
                return scipy.sparse.hstack(list(self.sp.values())).tocsr()
            elif len(sp_cols) == 0 and len(df_cols) > 0:
                # Take data from only df
                return self.df[df_cols]
            elif len(sp_cols) == 0 and len(df_cols) == 0:
                raise AttributeError(f'Columns {col} are not in DataTable.')
            else:
                # Cannot convert object cols to sparse!
                df_cols = [c for c in df_cols if str(self.df[c].dtype) != 'object']
                
                # Take data from both parts
                return scipy.sparse.hstack([self.df[df_cols].astype('float')] + list(self.sp.values())).tocsr()
        
        # If not a list, col must be a string
        if not isinstance(col, str):
            raise TypeError(
                    'Column name must be either a string or a list of strings.')
        
        # Check if column was duplicated
        if self.is_col_duplicate(col):
            raise AttributeError(
                    'Column exists in both sparse part and df part.')
            
        if col in self.df.columns:
            return self.df[col]
        elif col in self.sp:
            return self.sp[col]
        else:
            raise KeyError('Column does not exist.')
    
    def copy(self):
        # Create blank datatable
        new_dt = DataTable()
        
        # Copy dataframe part
        new_dt.df = self.df.copy()
        
        # Copy sparse part
        for col, vals in self.sp.items():
            new_dt.sp[col] = vals.copy()
        
        # update dt.columns
        new_dt.update_column_index()
        
        return new_dt
    
    def check_all_str(self, a):
        # Check if all elements are str
        if not all(map(lambda x: isinstance(x, str), a)):
            raise TypeError(
                    'All elements in columns must be strings')
    
    def get_cols_by_type(self, cols):
        # Get columns that exist in df
        df_cols = []
        sp_cols = []
        
        if len(self.df.columns) > 0:
            df_cols = [c for c in cols if c in self.df.columns]
            
        if len(self.sp) > 0:
            sp_cols = [c for c in cols if c in self.sp.keys()]
        
        for col in cols:
            if not (col in df_cols or col in sp_cols):
                raise KeyError(f'Column {col} does not exist in DataTable.')
        return df_cols, sp_cols
    
    def drop(self, cols):
        """Right now, always inplace.
        
        """
        
        if isinstance(cols, list):
            self.check_all_str(cols)
            df_cols, sp_cols = self.get_cols_by_type(cols)
            
            # Drop dataframe columns
            if len(df_cols) > 0:
                self.df.drop(df_cols, axis = 1, inplace = True)
            for col in sp_cols:
                del self.sp_cols[col]
        else:
            if not isinstance(cols, str):
                raise TypeError('column name must be either a string or a list of strings')
            else:
                if cols not in self.columns:
                    raise KeyError(f'column {col} does not exist.')
                else:
                    # Drop column
                    if col in self.df.columns:
                        # From dataframe
                        self.df.drop(col, axis = 1, inplace = True)
                        
                    else:
                        # From sparse
                        del self.sp[col]
        
    
    def __len__(self):
        return len(self.df)
        
    def __setitem__(self, col, val):
        if not isinstance(col, str):
            raise TypeError(
                    'Column name must be a string.')
        
        # If type is sparse, add to sp
        if isinstance(val, scipy.sparse.csr.csr_matrix):
            
            # If sparse, will add to sparse and drop df column
            if col in self.df.columns:
                self.df.drop(col, axis = 1, inplace = True)
            
            # Add/mutate
            self.sp[col] = val
            
        # If not sparse, add to df
        else:
            # If not sparse, will add to dataframe and drop sp column
            if col in self.sp:
                del self.sp[col]
                
            # Add/mutate
            self.df[col] = val
            
        # Update self.columns
        self.update_column_index()
            
    def data(self):
        #If there is a sparse part, return all data in sparse form
        if len(self.sp) > 0:
            return scipy.sparse.hstack([self.df] + list(self.sp.values())).tocsr()
        # If there is not sparse part, return df part
        else:
            return self.df
    
    def is_col_duplicate(self, col):
        # Check if col in dataframe part
        in_df = col in self.df.columns
        
        # Check if col in sparse part
        in_sparse = col in self.sp
        
        # If in both, column was duplicated
        return in_sparse and in_df
    
    def save(self, name):
        if not isinstance(name, str):
            raise TypeError('Filename must be a string.')
        
        # Form a dict from data
        file_dict = {'df': self.df,
                     'sp': self.sp}
        
        with open(name, 'wb') as fp:
            pickle.dump(file_dict, fp)
    
    def update_column_index(self):
        self.columns = pd.Index([])
        
        # Get dataframe columns
        if len(self.df.columns) > 0:
            self.columns = self.df.columns
            
        # Get sparse columns
        if len(self.sp) > 0:
            self.columns = self.columns.append(pd.Index(self.sp.keys()))
    
    def load(self, name):
        # Check if exists
        if os.path.exists(name):
            try:
                # Read pickle
                with open(name, 'rb') as fp:
                    file_dict = pickle.load(fp)
            except:
                print('Could not load data.')
            else:
                try:
                    # Get values
                    self.df = file_dict['df']
                    self.sp = file_dict['sp']
                    self.update_column_index()
                except:
                    print('Could not read data.')
        else:
            raise FileNotFoundError('Data does not exist.')
            
        