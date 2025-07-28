import os
import functools
import pandas as pd
import numpy as np


class TransformType:

    PRIMARY = 'primary'  # Takes primary datasource as input
    TRANSFORM = 'transform'  # Takes transformed datasource as input
    DUAL = 'dual'  # Takes both primary and transformed datasources as inputs


class TransformRegistry:
    
    """ Registry for transformation functions that can be composed together"""
    
    def __init__(self):
    
        """ Initialization """
        
        self._transforms = {}
        self._transform_types = {}
    
    def register(self, name, transform_func=None, transform_type=TransformType.TRANSFORM):
        
        """
        Register a transformation function
        
        Args:
            name: Unique name for the transform
            transform_func: Transformation function to register
            transform_type: The type of transform (PRIMARY, TRANSFORM, or DUAL - see above)
            
        Returns:
            The registered function
        """

        def decorator(func):
            if name in self._transforms:
                raise ValueError(f"Transform '{name}' already registered")
            
            # Validate the transform type
            if transform_type not in [TransformType.PRIMARY, TransformType.TRANSFORM, TransformType.DUAL]:
                raise ValueError(f"Invalid transform type: {transform_type}")
            
            self._transforms[name] = func
            self._transform_types[name] = transform_type
            return func
        
        # Handle both decorator and direct usage
        if transform_func is None:
            return decorator
        else:
            return decorator(transform_func)
    
    def register_primary(self, name, transform_func=None):

        """
        Register a PRIMARY transformation function (takes primary datasource only)

        """

        return self.register(name, transform_func, TransformType.PRIMARY)
    
    def register_transform(self, name, transform_func=None):

        """
        Register a TRANSFORM transformation function (takes transformed datasource only)
        """
        return self.register(name, transform_func, TransformType.TRANSFORM)
    
    def register_dual(self, name, transform_func=None):

        """

        Register a DUAL transformation function (takes both primary and transformed data)
        
        """
        return self.register(name, transform_func, TransformType.DUAL)
    
    def get(self, name):

        """
        Get a registered transformation function by name
        
        """
        if name not in self._transforms:
            raise KeyError(f"Transform '{name}' not found in registry")
            
        return self._transforms[name]
    
    def get_transform_type(self, name):

        """
        Get the type of a registered transform
        """
        if name not in self._transform_types:
            raise KeyError(f"Transform '{name}' not found in registry")
            
        return self._transform_types[name]
    
    def list_transforms(self):

        """
        List all registered transforms
      
        """
        return list(self._transforms.keys())
    
    def compose_with_feedback(self, transform_names, **kwargs):
        """
        Compose multiple transforms into a single function that correctly handles
        all transform types and maintains both primary and transformed datasources
        
        Args:
            transform_names: List of transform names to compose in order
            **kwargs: Additional keyword arguments to pass to all transforms
            
        Returns:
            Composed transformation function that returns (primary_data, derived_data)
        """
        # Get all the transform functions and their types
        transforms = []
        for name in transform_names:
            if name not in self._transforms:
                raise KeyError(f"Transform '{name}' not found in registry")
            transforms.append((name, self._transforms[name], self._transform_types[name]))
        
        # Define the composed function that applies transforms in sequence
        def composed_transform_with_feedback(data, **inner_kwargs):

            merged_kwargs = {**kwargs, **inner_kwargs}
            
            # Initial state: primary data only, no derived data yet
            primary_data = data
            derived_data = None
            
            # Apply each transform in sequence
            for i, (name, transform, transform_type) in enumerate(transforms):
                try:
                    if i == 0:
                        # First transform must accept primary data only
                        if transform_type == TransformType.TRANSFORM:
                            raise TypeError(f"First transform '{name}' cannot be of type TRANSFORM as there's no transformed data yet")
                            
                        # Apply the first transform to primary data
                        result = transform(primary_data, **merged_kwargs)
                    else:
                        # For subsequent transforms, handle based on type
                        if transform_type == TransformType.PRIMARY:
                            result = transform(primary_data, **merged_kwargs)
                        elif transform_type == TransformType.TRANSFORM:
                            result = transform(derived_data, **merged_kwargs)
                        elif transform_type == TransformType.DUAL:
                            result = transform(primary_data, derived_data, **merged_kwargs)
                        else:
                            raise ValueError(f"Unknown transform type: {transform_type}")
                except Exception as e:
                    raise RuntimeError(f"Error applying transform '{name}' ({transform_type}): {str(e)}") from e
                
                if isinstance(result, tuple) and len(result) == 2:
                    # Update both primary and derived data
                    primary_update, derived_update = result
                    if primary_update is not None and id(primary_update) != id(primary_data):
                        primary_data = primary_update
                    derived_data = derived_update
                else:
                    # Update only derived data
                    derived_data = result
            
            # Return the final state of both primary and derived data
            return primary_data, derived_data
        
        return composed_transform_with_feedback