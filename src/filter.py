import os
import pandas as pd
import numpy as np
import time

class Filter:

    """ Base filter class """
    
    def __init__(self, name):
        self.name = name
        self.enabled = True
        self.column = None
        self._filter_func = None  # lambda function
        self._validated = False
        self._last_mask = None  # cache the last computed mask

    def _create_filter_mask(self, df):
        """ Apply the lambda function to create the filter mask """
        if self._filter_func is None:
            raise RuntimeError(f"Filter {self.name} has no filter function defined")
        return self._filter_func(df)

    def _validate_dataframe(self, df):
        """ Validate the filter against the dataframe """
        if self.column not in df.columns:
            raise ValueError(f"Column '{self.column}' not found in DataFrame")

    def _validate(self, df):
        raise NotImplementedError    
    
    def get_filter_mask(self, df):
        """ Get the boolean mask for filtering """
        if not self._validated:
            self._validate(df)
        self._last_mask = self._create_filter_mask(df)
        return self._last_mask

    def apply_to_df(self, df):
        """ Apply filter to dataframe """
        if not self.enabled:
            return df, df.index.tolist()
        
        mask = self.get_filter_mask(df)
        filtered_df = df.loc[mask]
        return filtered_df, filtered_df.index.tolist()
    
    def apply_to_array(self, df, X):
        """ Apply filter to dataframe and use the corresponding
            mask to filter a numpy array """
        if not self.enabled:
            return df, X
        
        if X.shape[0] != len(df.index):
            raise ValueError("Number of rows in np array should equal number of rows in dataframe")
        
        # Try to reuse existing mask if possible
        if self._last_mask is not None and len(self._last_mask) == len(df):
            mask = self._last_mask
        else:
            # Compute new mask
            mask = self.get_filter_mask(df)
        
        # Apply mask to array
        filtered_array = X[mask]
        return df.loc[mask], filtered_array
    
    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False
    
    def clear_cache(self):
        """ Clear cached mask """
        self._last_mask = None

   
class CustomFilter(Filter):

    """
    Custom filter that accepts an arbitrary lambda function for filtering
    """
    
    def __init__(self, name, filter_func, columns=None):
        """
        Initialize a custom filter with an arbitrary filter function
        
        Args:
            name: Name of the filter
            filter_func: Lambda function that takes a pandas DataFrame and returns a boolean mask
            columns: List of column names used by the filter function (for validation)
                     If None, no column validation will be performed
        """
        super().__init__(name)
        self.filter_expr = filter_func
        self.columns = columns if columns is not None else []
        self._filter_func = filter_func  
        
    def _validate(self, df):
        """
        Validate that all required columns exist in the DataFrame
        """
        if not self.columns:
            # If no columns specified, skip validation
            self._validated = True
            return
            
        missing_columns = [col for col in self.columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Columns {missing_columns} not found in DataFrame")
            
        # debugging
        try:
            sample_df = df.head(1) if len(df) > 0 else df
            result = self._filter_func(sample_df)
            # Check if the result can be cast to a boolean array
            if len(sample_df) > 0 and not isinstance(result, (bool, np.ndarray)):
                raise ValueError("Filter function must return a boolean or boolean array")
        except Exception as e:
            raise ValueError(f"Filter function execution failed: {str(e)}")
            
        self._validated = True
    
    def __str__(self):
        if self.columns:
            return f"CustomFilter named {self.name} using columns {self.columns}"
        else:
            return f"CustomFilter named {self.name}"


class ThresholdFilter(Filter):

    """Threshold filter that selects df rows based on a set of thresholding operations"""
    
    _VALID_COMPARISONS = {"<", "<=", ">", ">=", "==", "!="}
    
    def __init__(self, name, column, threshold, comparison):
        super().__init__(name)
        if comparison not in self._VALID_COMPARISONS:
            raise ValueError(f"Must be one of {self._VALID_COMPARISONS}")
       
        self.column = column
        self.threshold = threshold
        self.comparison = comparison
        
        # Define the filter function as a lambda at initialization time
        self._define_filter_function()

    def _validate(self, df):
        self._validate_dataframe(df)
        self._validated = True
    
    def _define_filter_function(self):
        """Define the lambda function for this filter"""
        column = self.column  
        threshold = self.threshold 
        
        if self.comparison == "<":
            self._filter_func = lambda df: df[column].values < threshold
        elif self.comparison == ">":
            self._filter_func = lambda df: df[column].values > threshold
        elif self.comparison == "<=":
            self._filter_func = lambda df: df[column].values <= threshold
        elif self.comparison == ">=":
            self._filter_func = lambda df: df[column].values >= threshold
        elif self.comparison == "==":
            self._filter_func = lambda df: df[column].values == threshold
        elif self.comparison == "!=":
            self._filter_func = lambda df: df[column].values != threshold
            
    def __str__(self):
        return f"Threshold Filter named {self.name} : (df[{self.column}] {self.comparison} {self.threshold})"


class BooleanFilter(Filter):
    """Boolean filter that selects df rows based on a boolean flag"""
    
    def __init__(self, name, column, value):
        super().__init__(name) 
        self.column = column
        self.value = value
        
        # Define the filter function as a lambda at initialization time
        self._define_filter_function()

    def _validate(self, df):

        self._validate_dataframe(df)
        first_val = df[self.column].iloc[0]
        if not isinstance(first_val, (bool, np.bool_)):
            raise ValueError(f"Column '{self.column}' should be of boolean type")
        
        self._validated = True

    def _define_filter_function(self):

        column = self.column  
        value = self.value  
        
        if value:
            self._filter_func = lambda df: df[column].values
        else:
            self._filter_func = lambda df: ~df[column].values
        
    def __str__(self):
        return f"BooleanFilter named {self.name}: (df[{self.column}] is {self.value})"


class ValueFilter(Filter):
    
    """
    
    Value filter that selects df rows based on a set of values for a particular column
    i.e df[column_name].isin(values)

    """
    
    def __init__(self, name, column, values, include=True):
        super().__init__(name)
        self.column = column
        self.values = set(values) if not isinstance(values, set) else values
        self.include = include
        self._define_filter_function()

    def _validate(self, df):
        self._validate_dataframe(df)   
        if not set(df[self.column].unique()).intersection(self.values):
            raise ValueError(f"Column '{self.column}' does not contain values in {self.values}")
            
        self._validated = True
    
    def _define_filter_function(self):

        column = self.column  
        values = self.values 
        include = self.include 
        
        if include:
            self._filter_func = lambda df: df[column].isin(values).values
        else:
            self._filter_func = lambda df: ~df[column].isin(values).values
    
    def __str__(self):
        action = "in" if self.include else "not in"
        return f"ValueFilter named {self.name}: (df[{self.column}] {action} {self.values})"


class RangeFilter(Filter):
    """Range filter that selects df rows based on a values within a certain range"""
    
    def __init__(self, name, column, value_range, inclusive):
        super().__init__(name)
        
        if value_range[1] < value_range[0]:
            raise ValueError(f"min value {value_range[0]} has to be greater than max value {value_range[1]}")
        
        self.column = column
        self.value_range = value_range
        self.inclusive = inclusive
        self._define_filter_function()

    @classmethod
    def from_window(cls, name, column, center, window, inclusive):
    
        # Convert to higher precision to minimize rounding errors
        center = np.float64(center)
        left_window = np.float64(window[0])
        right_window = np.float64(window[1])
        
        min_value = center - abs(left_window)
        max_value = center + abs(right_window)
        

        if abs(min_value) < 1e-10:
            min_value = 0.0 
        
        decimals = 10  # Adjust based on precision needs
        min_value = np.round(min_value, decimals)
        max_value = np.round(max_value, decimals)
        return cls(name, column, (min_value, max_value), inclusive)
    
    def _validate(self, df):
        self._validate_dataframe(df)
        max_val = df[self.column].max()
        min_val = df[self.column].min()

        if self.value_range[0] < min_val or self.value_range[1] > max_val:
           print(f"WARNING: range provided may lie outside of range supported by {self.column}")
           
        self._validated = True
    
    def _define_filter_function(self):
        column = self.column 
        min_val, max_val = self.value_range  
        inclusive = self.inclusive 
        
        if inclusive[0] and inclusive[1]:
            self._filter_func = lambda df: (df[column].values >= min_val) & (df[column].values <= max_val)
        elif inclusive[0]:
            self._filter_func = lambda df: (df[column].values >= min_val) & (df[column].values < max_val)
        elif inclusive[1]:
            self._filter_func = lambda df: (df[column].values > min_val) & (df[column].values <= max_val)
        else:
            self._filter_func = lambda df: (df[column].values > min_val) & (df[column].values < max_val)
    
    def __str__(self):
        min_op = ">=" if self.inclusive[0] else ">"
        max_op = "<=" if self.inclusive[1] else "<"
        return f"RangeFilter named {self.name}: (df[{self.column}] {min_op} {self.value_range[0]} and {max_op} {self.value_range[1]})"
    
class PaddedRangeFilter(RangeFilter):
    
    """
    
    Enhanced RangeFilter that supports padded arrays with NaN values when
    parts of the specified range are missing from df.
    
    """
    
    def __init__(self, name, column, value_range, inclusive, pad_missing=True):

        super().__init__(name, column, value_range, inclusive)
        self.pad_missing = pad_missing
        
        self.min_val, self.max_val = value_range
        self.min_inclusive, self.max_inclusive = inclusive
        
        self._orig_indices = None
        self._index_map = None
        self._full_range_values = None
    
    @classmethod
    def from_window(cls, name, column, center, window, inclusive, pad_missing=True):

        # Convert to higher precision to minimize rounding errors
        center = np.float64(center)
        left_window = np.float64(window[0])
        right_window = np.float64(window[1])
        
        min_value = center - abs(left_window)
        max_value = center + abs(right_window)
        
        if abs(min_value) < 1e-10:
            min_value = 0.0

        decimals = 10  # Adjust based on your precision needs
        min_value = np.round(min_value, decimals)
        max_value = np.round(max_value, decimals)
        return cls(name, column, (min_value, max_value), inclusive, pad_missing=pad_missing)
    
    def _compute_full_range(self, df):
        """
        Computes the full range of values that should be in the result
        """
        if not self.pad_missing:
            # If not padding, just return None
            return None
        
        # Get the step size from the data (assumes uniform spacing)
        col_values = df[self.column].sort_values().unique()
        if len(col_values) < 2:
            # Can't determine step size with fewer than 2 values
            return col_values
            
        step_size = col_values[1] - col_values[0]
        
        start = self.min_val if self.min_inclusive else self.min_val + step_size
        end = self.max_val if self.max_inclusive else self.max_val - step_size
        
        decimals = -int(np.floor(np.log10(step_size))) + 6
        start = np.round(start, decimals)
        end = np.round(end, decimals)
        num_steps = int(np.round((end - start) / step_size)) + 1
        
        full_range = np.linspace(start, end, num_steps)
        return full_range
    
    def _build_index_mapping(self, df):

        """

        Build a mapping from the original data indices to the full range

        """
        if not self.pad_missing:
            # If not padding, don't build mapping
            return None, None
            
        # Get the full range of values
        full_range = self._compute_full_range(df)
        if full_range is None:
            # Couldn't compute full range
            return None, None
            
        # Create a mapping from data values to their target indices
        mask = self.get_filter_mask(df)
        filtered_df = df.loc[mask]
        filtered_values = filtered_df[self.column].values
        original_indices = filtered_df.index.values
        
        # Round values for comparison to avoid floating point issues
        decimals = -int(np.floor(np.log10(np.mean(np.diff(full_range))))) + 6
        rounded_full_range = np.round(full_range, decimals)
        rounded_filtered_values = np.round(filtered_values, decimals)
        
        # Create index mapping (value -> position in full range)
        index_map = {}
        for idx, val in zip(original_indices, rounded_filtered_values):
            try:
                pos = np.where(rounded_full_range == val)[0][0]
                index_map[idx] = pos
            except (IndexError, KeyError):
                # Value not in full range, shouldn't happen
                continue
                
        return index_map, full_range
    
    def apply_to_array(self, df, X):
        
        """
        
        Apply filter to dataframe and corresponding array, with optional padding
        This overrides the corresponding method in the base Filter class
        
        """
        if not self.enabled:
            return df, X
            
        if X.shape[0] != len(df.index):
            raise ValueError("Number of rows in array should equal number of rows in dataframe")
        
        # If padding is disabled, use the standard implementation
        if not self.pad_missing:
            return super().apply_to_array(df, X)
        
        mask = self.get_filter_mask(df)
        filtered_df = df.loc[mask]
        
        self._index_map, self._full_range_values = self._build_index_mapping(df)
        
        if self._index_map is None or self._full_range_values is None:
            # Couldn't build mapping, use standard implementation
            return filtered_df, X[mask]
        
        array_shape = list(X.shape)
        array_shape[0] = len(self._full_range_values)
        padded_array = np.full(array_shape, np.nan)
        
        for orig_idx, new_idx in self._index_map.items():
            if orig_idx in df.index:
                pos_in_orig = df.index.get_loc(orig_idx)
                if 0 <= new_idx < len(padded_array):
                    padded_array[new_idx] = X[pos_in_orig]
        
        return filtered_df, padded_array

    def __str__(self):
        pad_str = " with padding" if self.pad_missing else ""
        min_op = ">=" if self.inclusive[0] else ">"
        max_op = "<=" if self.inclusive[1] else "<"
        return f"PaddedRangeFilter named {self.name}: (df[{self.column}] {min_op} {self.value_range[0]} and {max_op} {self.value_range[1]}){pad_str}"



class FilterChain:
    
    """
    
    This is a container class that contains a set of (potentially different) filters
    and applies it in a serial fashion. For example, suppose D_{0} is our raw data
    then, D_{0} ---> D_{1} ---> D_{2} ---> ..... ---> D_{n}, where each arrow
    corresponds to a filter application

    """
    
    def __init__(self, name):
        self.filters = []
        self.name = name
        self._combined_func = None  # Combined filter function
        self._need_rebuild = True  # is rebuilding needed ?
        self._combined_mask = None  # Cache for the combined mask
        # ID of the last filtered DataFrame. This is to track filter chain state
        self._last_filtered_df_id = None  

    def add_filter(self, filt_obj):
        """Add a filter to the chain"""
        self.filters.append(filt_obj)
        self._need_rebuild = True
        self._combined_mask = None  # Invalidate mask cache

    def set_name(self, name):
        """Set the name of the filter chain"""
        self.name = name

    def reset(self):
        """Reset the filter chain"""
        self.filters = []
        self._combined_func = None
        self._need_rebuild = True
        self._combined_mask = None  # Invalidate mask cache
        self._last_filtered_df_id = None

    def __iter__(self):
        return iter(self.filters)
    
    def __getitem__(self, index):
        return self.filters[index]
    
    def __len__(self):
        return len(self.filters)
    
    def _get_df_id(self, df):
        """Get a unique identifier for a DataFrame to detect changes"""
        return (id(df), len(df))
    
    def _validate_all(self, df):
        """Validate all filters against the dataframe"""        
        if not self.filters:
            return
            
        err_msg = []
        for f in self.filters:
            if f.enabled and not f._validated:
                try:
                    f._validate(df)
                except Exception as e:
                    err_msg.append(f"{f.name}: {str(e)}")
        
        if err_msg:
            raise RuntimeError(f"Invalid filters in chain: {' ,'.join(err_msg)}")
            
    
    def _build_combined_function(self):
        """Build a combined filter function from all enabled filters"""
        # Get lambda functions from all enabled filters
        funcs = [f._filter_func for f in self.filters if f.enabled]
        
        if not funcs:
            # No enabled filters, return all True mask
            self._combined_func = lambda df: np.ones(len(df), dtype=bool)
        elif len(funcs) == 1:
            # Just one filter, use its function directly
            self._combined_func = funcs[0]
        else:
            # Combine multiple filter functions
            def combined_filter(df):
                # Start with all True mask
                mask = np.ones(len(df), dtype=bool)
                
                # Apply each filter function
                for func in funcs:
                    filter_mask = func(df)
                    mask &= filter_mask
                    
                    if not np.any(mask):
                        break
                        
                return mask
                
            self._combined_func = combined_filter
            
        self._need_rebuild = False
    
    def apply_to_df(self, df):

        """Apply filter chain to dataframe"""
        # No filters or no enabled filters
        if not self.filters or not any(f.enabled for f in self.filters):
            return df, df.index.tolist()
        
        # Check if we can reuse cached mask
        df_id = self._get_df_id(df)
        can_reuse_mask = (self._combined_mask is not None and 
                         self._last_filtered_df_id == df_id and
                         len(self._combined_mask) == len(df))
        
        if can_reuse_mask:
            mask = self._combined_mask
            result_df = df.loc[mask]
            return result_df, result_df.index.tolist()
        
        try:
            self._validate_all(df)
        except RuntimeError as e:
            raise e
        
        # Rebuild combined function if needed
        if self._need_rebuild or self._combined_func is None:
            self._build_combined_function()
        
        # Apply combined filter function
        mask = self._combined_func(df)
        result_df = df.loc[mask]
        
        # Cache the combined mask
        self._combined_mask = mask
        self._last_filtered_df_id = df_id
        
        return result_df, result_df.index.tolist()
    
    def apply_mask_to_array(self, array):

        if self._combined_mask is None:
            raise ValueError("No mask has been computed yet. Filter a DataFrame first.")
        
        if len(self._combined_mask) != len(array):
            raise ValueError(f"Mask length ({len(self._combined_mask)}) must match array length ({len(array)})")
        
        return array[self._combined_mask]
    
    def apply_to_array(self, df, array):
        """
        Apply filters to a DataFrame and then use the resulting mask to filter a numpy array
        
        Args:
            df: DataFrame to filter
            array: Numpy array with the same length as df to filter
            
        Returns:
            Tuple of (filtered_df, filtered_array)
        """
        if len(df) != len(array):
            raise ValueError(f"DataFrame length ({len(df)}) must match array length ({len(array)})")
        
        # Apply filters to DataFrame
        filtered_df, _ = self.apply_to_df(df)
        
        # Apply the same mask to the array
        filtered_array = self.apply_mask_to_array(array)
        
        return filtered_df, filtered_array
    
    def apply_filters_parallel(self, df):
        """
        Apply each filter separately to the dataframe
        i.e applies filters in parallel instead of serially

        Args:
            df: DataFrame to filter
            
        Returns:
            List of tuples (filtered_df, indices) for each enabled filter
        """
        # Skip if no filters
        if not self.filters or not any(f.enabled for f in self.filters):
            return [(df, df.index.tolist())]
        
        # Validate all filters
        try:
            self._validate_all(df)
        except RuntimeError as e:
            raise e
        
        # Apply each enabled filter separately
        results = []
        for filt in self.filters:
            if filt.enabled:
                filtered_df, indices = filt.apply_to_df(df)
                results.append((filtered_df, indices))
        
        return results

    def apply_to_array_parallel(self, df, array):
        """
        Apply each filter separately to a DataFrame and corresponding array
        i.e applies filters in parallel instead of serially

        Args:
            df: DataFrame to filter
            array: Numpy array with the same length as df
            
        Returns:
            List of tuples (filtered_df, filtered_array) for each enabled filter
        """
        if len(df) != len(array):
            raise ValueError(f"DataFrame length ({len(df)}) must match array length ({len(array)})")
        
        # Skip if no filters
        if not self.filters or not any(f.enabled for f in self.filters):
            return [(df, array)]
        
        # Validate all filters
        try:
            self._validate_all(df)
        except RuntimeError as e:
            raise e
        
        # Apply each enabled filter separately
        results = []
        for filt in self.filters:
            if filt.enabled:
                filtered_df, filtered_array = filt.apply_to_array(df, array)
                results.append((filtered_df, filtered_array))
        
        return results
    

    def apply_to_array_parallel_fast(self, df, array):
       
        """
        A fast version of filter application to a df and np array
        
        Args:
            df: DataFrame to filter
            array: Numpy array with the same length as df
            
        Returns:
            LazyArrayFilterResults object (see def below)

        """
        if len(df) != len(array):
            raise ValueError(f"DataFrame length ({len(df)}) must match array length ({len(array)})")
        
        if not self.filters or not any(f.enabled for f in self.filters):
            return [(df, array)]
        
        try:
            self._validate_all(df)
        except RuntimeError as e:
            raise e
        
        filter_masks = {}
        enabled_filters = []
        
        for filt in self.filters:
            if filt.enabled:
                enabled_filters.append(filt)
                # Only compute mask if not already cached
                if not hasattr(filt, '_last_mask') or filt._last_mask is None or len(filt._last_mask) != len(df):
                    filt.get_filter_mask(df)  # This updates filt._last_mask
                filter_masks[filt.name] = filt._last_mask
        
        results = []
        for filt in enabled_filters:
            mask = filter_masks[filt.name]
            results.append((filt, mask))
        
        return LazyArrayFilterParallel(results, df, array)
    
    def clear_cache(self):
        """Clear cached masks"""
        self._combined_mask = None
        self._last_filtered_df_id = None
        for filt in self.filters:
            if hasattr(filt, 'clear_cache'):
                filt.clear_cache()


class LazyArrayFilterParallel:
    """
    Class for handling multiple filtered arrays with proper
    support for PaddedRangeFilters.
    """
    
    def __init__(self, lazy_results, df, array):
        """
        Initialize with minimal computation for maximum speed
        
        Args:
            lazy_results: List of (filter, mask) tuples
            df: The source DataFrame
            array: The source array to filter
        """
        self.lazy_results = lazy_results
        self.df = df
        self.array = array
        self.cached_results = {}
        
        # Extract filters and masks with minimal processing
        self._filters = [f for f, _ in lazy_results]
        self._masks = [m for _, m in lazy_results]
        
        self._filtered_sizes = [np.sum(mask) for mask in self._masks]
        self._min_filtered_size = min(self._filtered_sizes) if self._filtered_sizes else 0
        self._max_filtered_size = max(self._filtered_sizes) if self._filtered_sizes else 0
        
        self._uniform_size = len(set(self._filtered_sizes)) == 1 if self._filtered_sizes else True
        
        self._has_padded_filters = any(
            isinstance(f, PaddedRangeFilter) and f.pad_missing 
            for f in self._filters
        )
    
    def __len__(self):
        """Return the number of filter results available"""
        return len(self.lazy_results)
    
    def __getitem__(self, idx):
        """Get a specific filter result (with lazy evaluation)"""
        if isinstance(idx, slice):
            indices = list(range(*idx.indices(len(self))))
            return [self[i] for i in indices]
            
        if idx in self.cached_results:
            return self.cached_results[idx]
            
        filt, mask = self.lazy_results[idx]
        
        if isinstance(filt, PaddedRangeFilter) and filt.pad_missing:
            filtered_df, filtered_array = filt.apply_to_array(self.df, self.array)
        else:
            filtered_df = self.df.loc[mask]
            filtered_array = self.array[mask]
            
        result = (filtered_df, filtered_array)
        
        self.cached_results[idx] = result
        return result
    
    def __iter__(self):
        """Iterate through all filter results"""
        for i in range(len(self)):
            yield self[i]
    
    def get_subset(self, indices):
        """Get only a specific subset of filter results"""
        return [self[i] for i in indices]
    
    def get_filter_names(self):
        """Get the names of all filters"""
        return [f.name for f in self._filters]
    
    def get_filtered_sizes(self):
        """Get the number of rows in each filtered output"""
        return self._filtered_sizes
    
    def has_uniform_size(self):
        return self._uniform_size
    
    def get_filtered_arrays_only(self, indices=None):
        if indices is None:
            indices = range(len(self))
            
        arrays = []
        for i in indices:
            _, filtered_array = self[i]
            arrays.append(filtered_array)
            
        return arrays
    
    def get_stacked_arrays(self, indices=None, truncate=False, min_size=None):
        
        """
        Optimized implementation that correctly handles PaddedRangeFilters
        while maintaining high speed.
        
        """
        start_time = time.time()
        
        if indices is None:
            indices = list(range(len(self.lazy_results)))
        else:
            indices = list(indices)
        
        # Quick validation of indices
        valid_indices = [i for i in indices if i < len(self.lazy_results)]
        if not valid_indices:
            return None
        
        # For PaddedRangeFilters, we need to handle each filter differently
        if self._has_padded_filters:
            
            # First, identify which filters are PaddedRangeFilters and which are regular
            padded_indices = []
            regular_indices = []
            arrays = [None] * len(valid_indices) 
            
            for idx, i in enumerate(valid_indices):
                filt = self.lazy_results[i][0]
                if isinstance(filt, PaddedRangeFilter) and filt.pad_missing:
                    padded_indices.append((idx, i))
                else:
                    regular_indices.append((idx, i))
            
            # Process regular filters in batches (much faster)
            if regular_indices:
                for idx, i in regular_indices:
                    if i in self.cached_results:
                        _, arr = self.cached_results[i]
                    else:
                        _, mask = self.lazy_results[i]
                        arr = self.array[mask]
                        self.cached_results[i] = (self.df.loc[mask], arr)
                    
                    arrays[idx] = arr
            
            # Process padded filters in batches
            if padded_indices:
                for idx, i in padded_indices:
                    if i in self.cached_results:
                        # Use cached result
                        _, arr = self.cached_results[i]
                    else:
                        # Use filter's apply_to_array method
                        filt, _ = self.lazy_results[i]
                        _, arr = filt.apply_to_array(self.df, self.array)
                        # Cache the result
                        self.cached_results[i] = (None, arr)  # We'll fill in df later if needed
                    
                    arrays[idx] = arr
            
            should_truncate = truncate
            if truncate is None:
                array_lengths = [len(arr) for arr in arrays]
                should_truncate = len(set(array_lengths)) > 1
            
            if should_truncate:
                array_lengths = [len(arr) for arr in arrays]
                trunc_size = min(array_lengths) if array_lengths else 0
                if min_size is not None:
                    trunc_size = min(trunc_size, min_size)
                
                if trunc_size == 0:
                    return None
                
                for i in range(len(arrays)):
                    if len(arrays[i]) > trunc_size:
                        arrays[i] = arrays[i][:trunc_size]
        else:
            array_shapes = set()
            need_shape_check = True
            
            sizes = [self._filtered_sizes[i] for i in valid_indices if i < len(self._filtered_sizes)]
            if not sizes:
                return None
                
            should_truncate = truncate
            if truncate is None:
                should_truncate = not self._uniform_size
            
            if should_truncate:
                trunc_size = min(sizes)
                if min_size is not None:
                    trunc_size = min(trunc_size, min_size)
                
                if trunc_size == 0:
                    return None
                    
                if self.array.ndim > 1:
                    if all(s >= trunc_size for s in sizes):
                        need_shape_check = False
                        result_shape = (len(valid_indices), trunc_size, self.array.shape[1])
                        try:
                            result = np.empty(result_shape)
                            
                            for idx, i in enumerate(valid_indices):
                                # Get mask
                                _, mask = self.lazy_results[i]
                                # Get indices where mask is True
                                indices_to_use = np.where(mask)[0][:trunc_size]
                                # Fill directly into result array
                                result[idx] = self.array[indices_to_use]
                            
                            return result
                        except:
                            need_shape_check = True
            
            if need_shape_check:
                arrays = []
                for i in valid_indices:
                    if i in self.cached_results:
                        _, arr = self.cached_results[i]
                    else:
                        _, mask = self.lazy_results[i]
                        arr = self.array[mask]
                        self.cached_results[i] = (self.df.loc[mask], arr)
                    
                    if should_truncate and trunc_size is not None and len(arr) > trunc_size:
                        arr = arr[:trunc_size]
                    
                    arrays.append(arr)
                    
                    array_shapes.add(arr.shape)
        
        try:
            if self._has_padded_filters:
                stacked = np.stack(arrays)
            else:
                if len(array_shapes) == 1:
                    stacked = np.stack(arrays)
                else:
                    stacked = np.array(arrays, dtype=object)
            
            return stacked
        except ValueError:

            return np.array(arrays, dtype=object)

    
    def compute_mean_across_filters(self, truncate=None):
        """
        Compute the mean across all filters
        
        Args:
            truncate: True to force truncation, False to force no truncation,
                     None to auto-detect based on array sizes (default)
            
        Returns:
            Numpy array with the mean values across all filters
        """
        if len(self.lazy_results) == 0:
            return None
        
        # Try to use stacked arrays method for efficiency
        try:
            stacked = self.get_stacked_arrays(truncate=truncate)
            if stacked is not None and stacked.ndim > 1:
                # For padded filters, use nanmean to handle NaN values
                if self._has_padded_filters:
                    return np.nanmean(stacked, axis=0)
                else:
                    return np.mean(stacked, axis=0)
        except:
            pass
        
        filtered_arrays = self.get_filtered_arrays_only()
        
        if not filtered_arrays:
            return None
        
        use_truncation = truncate
        if truncate is None:
            use_truncation = not self._uniform_size
        
        if use_truncation:
            min_size = min(len(arr) for arr in filtered_arrays)
            if min_size == 0:
                return None
            
            truncated_arrays = [arr[:min_size] for arr in filtered_arrays]
            filtered_arrays = truncated_arrays
        
        # Compute mean
        if self._has_padded_filters:
            # For PaddedRangeFilters, use nanmean to handle NaN values
            if self.array.ndim > 1:
                try:
                    stacked = np.stack(filtered_arrays)
                    return np.nanmean(stacked, axis=0)
                except:
                    d_features = self.array.shape[1]
                    col_sums = np.zeros(d_features)
                    col_counts = np.zeros(d_features)
                    
                    for arr in filtered_arrays:
                        for col in range(d_features):
                            if col < arr.shape[1]:
                                col_values = arr[:, col]
                                valid_mask = ~np.isnan(col_values)
                                col_sums[col] += np.sum(col_values[valid_mask])
                                col_counts[col] += np.sum(valid_mask)
                    
                    return np.divide(col_sums, col_counts, 
                                  out=np.zeros_like(col_sums), 
                                  where=col_counts>0)
            else:
                total_sum = 0
                total_count = 0
                
                for arr in filtered_arrays:
                    valid_mask = ~np.isnan(arr)
                    total_sum += np.sum(arr[valid_mask])
                    total_count += np.sum(valid_mask)
                
                return total_sum / total_count if total_count > 0 else np.nan
        else:
            if self.array.ndim > 1:
                try:
                    stacked = np.stack(filtered_arrays)
                    return np.mean(stacked, axis=0)
                except:
                    d_features = self.array.shape[1]
                    col_sums = np.zeros(d_features)
                    col_counts = np.zeros(d_features)
                    
                    for arr in filtered_arrays:
                        for col in range(d_features):
                            if col < arr.shape[1]:
                                col_values = arr[:, col]
                                col_sums[col] += np.sum(col_values)
                                col_counts[col] += len(col_values)
                    
                    return np.divide(col_sums, col_counts, 
                                  out=np.zeros_like(col_sums), 
                                  where=col_counts>0)
            else:
                total_sum = sum(np.sum(arr) for arr in filtered_arrays)
                total_count = sum(len(arr) for arr in filtered_arrays)
                return total_sum / total_count if total_count > 0 else np.nan