import pandas as pd
from src.config import TICKET_PRICES, FEATURE_COLUMNS


class TitanicPreprocessor:
    # define class variables for mappings
    sex_mapping = {"male": 0, "female": 1}
    embarked_mapping = {"C": 0, "Q": 1, "S": 2}
    economic_status_mapping = {"Low": 0, "Middle": 1, "Wealthy": 2}
    age_group_mapping = {"Child": 0, "Teenager": 1, "Adult": 2, "Senior": 3}

    def __init__(self):
        """Initialize mappings and placeholder for future attributes"""
        pass

    def _classify_status(
        self, row
    ):  # underscore is used to indicate that this method is private
        """Assign economic status based on Pclass & Fare."""
        if row["Pclass"] == 1 and row["Fare"] > 150:
            return "Wealthy"
        elif row["Pclass"] == 2 or (row["Pclass"] == 1 and row["Fare"] <= 150):
            return "Middle"
        else:
            return "Low"

    def _fill_missing_age(self, df):
        """Fill missing Age values using the median based on (Pclass, Sex) groups."""
        df["Age"] = df.groupby(["Pclass", "Sex"])["Age"].transform(
            lambda x: x.fillna(x.median())
        )
        return df

    def _categorize_age(self, age):
        if age <= 12:
            return "Child"
        elif age <= 19:
            return "Teenager"
        elif age <= 59:
            return "Adult"
        else:
            return "Senior"

    def _is_alone(self, row):
        """Determine if a passenger is traveling alone (family_size == 1)."""
        return 1 if row["family_size"] == 1 else 0

    def _manual_encode(self, df):
        """Apply manual mappings to categorical columns."""
        df["Sex"] = df["Sex"].map(TitanicPreprocessor.sex_mapping)  # use class variable
        df["Embarked"] = df["Embarked"].map(TitanicPreprocessor.embarked_mapping)
        df["economic_status"] = df["economic_status"].map(
            TitanicPreprocessor.economic_status_mapping
        )
        df["age_group"] = df["age_group"].map(TitanicPreprocessor.age_group_mapping)
        return df

    def transform(self, df):
        """Apply preprocessing steps to Titanic dataset."""

        # Drop unnecessary columns
        df = df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)

        # Remove rows with missing Embarked
        df = df[df["Embarked"].notna()]

        # Remove rows where Fare is missing or 0
        df = df[df["Fare"].notna() & (df["Fare"] > 0)]

        # Fill missing Age values
        df = self._fill_missing_age(df)

        # Create family size feature
        df["family_size"] = df["SibSp"] + df["Parch"] + 1

        # Create is_alone feature
        df["is_alone"] = df.apply(self._is_alone, axis=1)

        # Assign economic status
        df["economic_status"] = df.apply(self._classify_status, axis=1)

        # Categorize Age
        df["age_group"] = df["Age"].apply(self._categorize_age)

        # Apply manual mappings
        df = self._manual_encode(df)

        # Drop unnecessary columns
        df = df.drop(["SibSp", "Parch"], axis=1)

        # Standardize column names
        column_renaming = {
            "Survived": "survived",
            "Pclass": "class",
            "Sex": "sex",
            "Age": "age",
            "Fare": "fare",
            "Embarked": "embarked",
            "family_size": "family_size",
            "is_alone": "is_alone",
            "economic_status": "economic_status",
            "age_group": "age_group",
        }

        df = df.rename(columns=column_renaming)

        # Reset index
        df = df.reset_index(drop=True)

        return df  # Return the modified DataFrame


class InputPreprocessor:
    """
    Handles preprocessing of user input data for Titanic survival prediction
    """

    # Define class-level mappings
    sex_mapping = {"male": 0, "female": 1}
    embarked_mapping = {"cherbourg": 0, "queenstown": 1, "southampton": 2}
    economic_status_mapping = {"low": 0, "middle": 1, "wealthy": 2}
    age_group_mapping = {"child": 0, "teenager": 1, "adult": 2, "senior": 3}

    def __init__(
        self,
        name: str,
        gender: str,
        embarkation: str,
        family_members: int,
        age: int,
        ticket_class: int,
        ticket_type: str,
    ):
        """
        Initialize InputPreprocessor with user input

        Args:
            name: Passenger name
            gender: Male or Female
            embarkation: Port of embarkation
            family_members: Number of family members
            age: Passenger age
            ticket_class: Class of ticket (1, 2, or 3)
            ticket_type: Type of ticket (Budget, Standard, Premium, Luxury)
        """
        self.name = name
        self.gender = gender.lower()
        self.embarkation = embarkation.lower()
        self.family_members = family_members
        self.age = age
        self.ticket_class = ticket_class
        self.ticket_type = ticket_type

        # Validate inputs
        self._validate_inputs()

    def _validate_inputs(self):
        """Validate input data"""
        if self.gender not in self.sex_mapping:
            raise ValueError(
                f"Invalid gender. Must be one of {list(self.sex_mapping.keys())}"
            )

        if self.embarkation not in self.embarked_mapping:
            raise ValueError(
                f"Invalid embarkation. Must be one of {list(self.embarked_mapping.keys())}"
            )

        if not 0 <= self.age <= 100:
            raise ValueError("Age must be between 0 and 100")

        if not 1 <= self.ticket_class <= 3:
            raise ValueError("Ticket class must be 1, 2, or 3")

    def _classify_ticket_status(self) -> str:
        """Determine economic status based on ticket class and type"""
        if self.ticket_class == 1 and self.ticket_type == "Luxury":
            return "wealthy"
        elif self.ticket_class == 2 or (
            self.ticket_class == 1 and self.ticket_type != "Luxury"
        ):
            return "middle"
        else:
            return "low"

    def _categorize_age(self) -> str:
        """Categorize age into groups"""
        if self.age <= 12:
            return "child"
        elif self.age <= 19:
            return "teenager"
        elif self.age <= 59:
            return "adult"
        else:
            return "senior"

    def _is_alone(self) -> int:
        """Determine if passenger is traveling alone"""
        return 1 if self.family_members == 0 else 0

    def _calculate_family_size(self) -> int:
        """Calculate total family size including the passenger"""
        return self.family_members + 1

    def _get_fare(self) -> float:
        """Get fare based on ticket class and type"""
        return TICKET_PRICES.get((self.ticket_class, self.ticket_type), 0)

    def preprocess(self) -> pd.DataFrame:
        """
        Transform user input into model-compatible format

        Returns:
            DataFrame with preprocessed features
        """
        user_data = {
            "class": self.ticket_class,
            "sex": self.sex_mapping[self.gender],
            "age": self.age,
            "fare": self._get_fare(),
            "embarked": self.embarked_mapping[self.embarkation],
            "family_size": self._calculate_family_size(),
            "is_alone": self._is_alone(),
            "economic_status": self.economic_status_mapping[
                self._classify_ticket_status()
            ],
            "age_group": self.age_group_mapping[self._categorize_age()],
        }

        # Create DataFrame with correct column order
        return pd.DataFrame([user_data])[FEATURE_COLUMNS]

    @classmethod
    def prepare_user_input(
        cls,
        name: str,
        gender: str,
        embarkation: str,
        family_members: int,
        age: int,
        ticket_class: int,
        ticket_type: str,
    ) -> pd.DataFrame:
        """
        Factory method to create and preprocess user input

        Args:
            name: Passenger name
            gender: Male or Female
            embarkation: Port of embarkation
            family_members: Number of family members
            age: Passenger age
            ticket_class: Class of ticket (1, 2, or 3)
            ticket_type: Type of ticket (Budget, Standard, Premium, Luxury)

        Returns:
            DataFrame with preprocessed features
        """
        processor = cls(
            name, gender, embarkation, family_members, age, ticket_class, ticket_type
        )
        return processor.preprocess()


# Test the InputPreprocessor class
user_input = InputPreprocessor.prepare_user_input(
    name="John Doe",
    gender="Male",
    embarkation="Southampton",
    family_members=2,
    age=100,
    ticket_class=1,
    ticket_type="Luxury",
)

print(pd.DataFrame(user_input))
