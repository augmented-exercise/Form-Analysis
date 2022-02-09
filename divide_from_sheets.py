#! /usr/bin/env python3
"""
Grab information from google sheets before performing rep analysis
"""
from __future__ import print_function
from audioop import reverse

import os.path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

import repanalysis

# If modifying these scopes, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']

# The ID and range of a sample spreadsheet.
SPREADSHEET_ID = '11nKJtwIokTGkettgl19yj66LEkFy-ET4siLvePGNgRA'
RANGE_NAME = 'Timestamps!A1:G38'
REVERSE_RANGE = 'Exercises!A1:B6'

def get_data():
    """
    Gets data from the spreadsheet
    """
    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    try:
        service = build('sheets', 'v4', credentials=creds)

        # Call the Sheets API
        sheet = service.spreadsheets()
        result = sheet.values().get(spreadsheetId=SPREADSHEET_ID,
                                    range=RANGE_NAME).execute()
        reverse_result = sheet.values().get(spreadsheetId=SPREADSHEET_ID,
                                            range=REVERSE_RANGE).execute()

        values = result.get('values', [])
        reverse_values = reverse_result.get('values', [])

        if not values:
            print('No data found.')
            return

        return values, reverse_values
    except HttpError as err:
        print(err)

def main():
    """
    Read google spreadsheet and split up valid exercises into their reps
    """
    values, reverse_values = get_data()
    data_dir = "data"
    files = os.listdir(data_dir)

    reversed = {}
    print(reverse_values)
    for row in reverse_values[1:]:
        exercise, reverse = row
        reversed[exercise] = bool(reverse)
    
    for row in values[1:]:
        time, notes, valid, subject, form, exercise, name = row
        if valid == 'FALSE':
            continue
        # Get infile
        candidates = list(filter(lambda x : time in x, files))
        if len(candidates) > 1:
            print(f"Duplicate files for {time}")
            continue
        if len(candidates) == 0:
            print(f"No files found for {time}")
            continue
        infile = os.path.join(data_dir,candidates[0])
        print(infile)
        try:
            repanalysis.divide(infile, exercise, name, subject, reversed[exercise])
        except:
            print(f"Problem with {time}. This is {exercise} with {name} for {subject}")
if __name__ == '__main__':
    main()
    print("Done!")