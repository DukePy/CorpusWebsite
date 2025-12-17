EDUCATIONAL RESOURCES DIRECTORY
================================

This directory contains downloadable resources for educators.

CURRENT FILES (PLACEHOLDERS):
-----------------------------
1. Lesson-Plan-Cultural-Metaphors.pdf
2. Activity-Linguistic-Innovations.pdf
3. Guide-Translanguaging-Strategies.pdf

INSTRUCTIONS FOR UPDATING:
--------------------------
To replace these placeholder files with actual content:

1. Create your educational resource (PDF, Word doc, etc.)
2. Save it with the same filename as the placeholder
3. Replace the placeholder file in this directory
4. The download links on the website will automatically use the new file

FILE NAMING CONVENTION:
-----------------------
- Use hyphens (-) instead of spaces
- Keep names descriptive but concise
- Supported formats: PDF, DOCX, PPTX, ZIP

ADDING NEW RESOURCES:
---------------------
To add a new resource:

1. Place the file in this directory
2. Edit templates/index.html
3. Add a new resource item in the Educator's Hub section
4. Use the Flask url_for template:
   {{ url_for('static', filename='resources/Your-File-Name.pdf') }}

Example:
--------
<a href="{{ url_for('static', filename='resources/New-Resource.pdf') }}" download>
    Download
</a>

