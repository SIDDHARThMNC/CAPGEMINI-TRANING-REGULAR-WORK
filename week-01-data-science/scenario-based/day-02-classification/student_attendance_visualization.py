import matplotlib.pyplot as plt

months = ['January', 'February', 'March', 'April']
attendance = [85, 90, 95, 88]

print("=" * 70)
print("STUDENT ATTENDANCE ANALYSIS")
print("=" * 70)
print()

print("Monthly Attendance Data:")
for month, count in zip(months, attendance):
    print(f"{month}: {count} students")
print()

plt.figure(figsize=(10, 6))
plt.plot(months, attendance, marker='o', linewidth=2, markersize=8, color='blue')
plt.xlabel('Month', fontsize=12)
plt.ylabel('Number of Students Present', fontsize=12)
plt.title('Student Attendance Trend (First Semester)', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('student_attendance_chart.png', dpi=300, bbox_inches='tight')
plt.show()

print("=" * 70)
print("CHART INTERPRETATION")
print("=" * 70)
print()

highest_month = months[attendance.index(max(attendance))]
highest_count = max(attendance)
print(f"Highest Attendance: {highest_month} with {highest_count} students")
print()

print("Attendance Trend Analysis:")
print("- January to March: Steady increase (85 → 90 → 95)")
print("- March to April: Decline (95 → 88)")
print("- Overall Pattern: Upward trend with a dip in April")
print()

print("Possible Reasons for April Dip:")
print("- Mid-semester exams causing stress and absences")
print("- Spring break or holidays affecting attendance")
print("- Weather changes (seasonal illnesses)")
print("- Student fatigue after continuous months of classes")
print("- Assignment deadlines causing students to skip classes")
print()

april_drop = attendance[2] - attendance[3]
print(f"April Drop: {april_drop} students (from March)")
print(f"Percentage Drop: {(april_drop/attendance[2])*100:.1f}%")
print()

print("Chart saved as 'student_attendance_chart.png'")
