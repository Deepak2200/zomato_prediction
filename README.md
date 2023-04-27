class DataTransformationconfig:
    preprocessor_obj_file_path=os.path.join("artifacts","preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationconfig()

    def get_data_transformation_object(self):
        try:
            logging.info("Data transformation initiat")
            #define which columns should be ordinal-encoded and which should be scaled
            categorical_col=['Weather_conditions', 'Road_traffic_density', 'Type_of_order','Type_of_vehicle', 
                             'Festival', 'City']

            numerical_col=['Delivery_person_Age', 'Delivery_person_Ratings', 'Restaurant_latitude',
                           'Restaurant_longitude', 'Delivery_location_latitude','Delivery_location_longitude',
                             'Vehicle_condition','multiple_deliveries', 'order_day', 'order_month', 
                             'order_hours','order_min', 'picked_hours', 'picked_min']
            
            #define custom ranking of ordianl variable
            wheather=['Sandstorms','Fog', 'Stormy', 'Windy', 'Cloudy', 'Sunny']
            road_jam=['Jam', 'High', 'Medium', 'Low']
            food_order=['Drinks', 'Buffet','Snack', 'Meal']
            vehical_cat=["bicycle","electric_scooter","scooter","motorcycle"]
            festival=["No","Yes"]
            city_cat=['Metropolitian', 'Urban', 'Semi-Urban']

            logging.info("pipline initiated")
            ## Numerical Pipeline
            num_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='median')),
                ('scaler',StandardScaler())

                ]

            )

            # Categorigal Pipeline
            cat_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('ordinalencoder',OrdinalEncoder(categories=[wheather,road_jam,food_order,vehical_cat,festival,city_cat])),
                ('scaler',StandardScaler())
                ]

            )

            preprocessor=ColumnTransformer([
            ('num_pipeline',num_pipeline,numerical_col),
            ('cat_pipeline',cat_pipeline,categorical_col)
            ])

            return preprocessor
        
            logging.info("pipeline completed")

        except Exception as e:
            logging.info("Error in Data Transformation")
            raise CustomException(e,sys)
        
    
    def initiate_data_transformation(self,train_path,test_path):
        try:
            #reading data train and test
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            
            logging.info("Read train and test data completed")
            logging.info(f"Train DataFrame head:\n{train_df.head().to_string()}")
            logging.info(f"Test DataFrame head:\n{test_df.head().to_string()}")

            logging.info("Obtaining preprocessor object")
            preprocessor_obj=self.get_data_transformation_object()

            #we are looking for train and test data
            target_column="Time_taken (min)"
            drop_columns=[target_column,"ID","Delivery_person_ID"]

            input_feature_train_df=train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df=train_df[target_column]

            input_feature_test_df=test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df=test_df[target_column]

            #Transforming using preprocessor obj
            input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessor_obj.transform(input_feature_test_df)
            
            logging.info("applying preprocessing object on training and testing dataset")

            train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )
            logging.info("preprocessor pickle file saved")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            logging.info("Exception occured in the initiate_data_transform")
            raise CustomException(e,sys)



obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()
    data_transformation=DataTransformation()


    class CustomData:
    def __init__(self,
                 Delivery_person_Age:float,
                 Delivery_person_Ratings:float,
                 Restaurant_latitude:float,
                 Restaurant_longitude:float,
                 Delivery_location_latitude:float,
                 Delivery_location_longitude:float,
                 Weather_conditions:str,
                 Road_traffic_density:str,
                 Vehicle_condition:int,
                 Type_of_order:str,
                 Type_of_vehicle:str,
                 multiple_deliveries:float,
                 Festival:str,
                 City:str,
                 order_day:float,
                 order_month:float,
                 order_hours:float,
                 order_min:float,
                 picked_hours:float,
                 picked_min:float):
                 
        self.Delivery_person_Age=Delivery_person_Age
        self.Delivery_person_Ratings=Delivery_person_Ratings
        self.Restaurant_latitude=Restaurant_latitude
        self.Restaurant_longitude=Restaurant_longitude
        self.Delivery_location_latitude=Delivery_location_latitude
        self.Delivery_location_longitude=Delivery_location_longitude
        self.Weather_conditions=Weather_conditions
        self.Road_traffic_density=Road_traffic_density
        self.Vehicle_condition=Vehicle_condition
        self.Type_of_order=Type_of_order
        self.Type_of_vehicle=Type_of_vehicle
        self.multiple_deliveries=multiple_deliveries
        self.Festival=Festival
        self.City=City
        self.order_day=order_day
        self.order_month=order_month
        self.order_hours=order_hours
        self.order_min=order_min
        self.picked_hours=picked_hours
        self.picked_min=picked_min
        
    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'Delivery_person_Age':[self.Delivery_person_Age],
                "Delivery_person_Ratings":[self.Delivery_person_Ratings],
                "Restaurant_latitude":[self.Restaurant_latitude],
                "Restaurant_longitude":[self.Restaurant_longitude],
                "Delivery_location_latitude":[self.Delivery_location_latitude],
                "Delivery_location_longitude":[self.Delivery_location_longitude],
                "Weather_conditions":[self.Weather_conditions],
                "Road_traffic_density":[self.Road_traffic_density],
                "Vehicle_condition":[self.Vehicle_condition],
                "Type_of_order":[self.Type_of_order],
                "Type_of_vehicle":[self.Type_of_vehicle],
                "multiple_deliveries":[self.multiple_deliveries],
                "Festival":[self.Festival],
                "City":[self.City],
                "order_day":[self.order_day],
                "order_month":[self.order_month],
                "order_hours":[self.order_hours],
                "order_min":[self.order_min],
                "picked_hours":[self.picked_hours],
                "picked_min":[self.picked_min]

                
                
            }